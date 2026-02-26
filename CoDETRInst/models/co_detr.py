import copy
from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig, ConfigType
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.models.task_modules.samplers import SamplingResult


@MODELS.register_module()
class CoDETRInst(BaseDetector):
    """用于实例分割的 Co-DETR 检测器。

    Args:
        backbone (dict): 主干网络的配置。
        neck (dict, optional): 颈部的配置。默认为 None。
        query_head (dict, optional): DETR 查询头的配置。默认为 None。
        mask_roi_extractor (dict, optional): 用于从特征图中提取 mask ROI 的配置。
            默认为 None。
        mask_head (dict, optional): mask 头的配置。默认为 None。
        mask_iou_head (dict, optional): 用于预测 mask 和真实值之间 IoU 的 mask IoU 头的配置。
            默认为 None。
        rpn_head (dict, optional): 两阶段 RPN 的配置。默认为 None。
        roi_head (list[dict], optional): 两阶段 ROI 头的配置列表。默认为 [None]。
        bbox_head (list[dict], optional): 一阶段 bbox 头的配置列表。默认为 [None]。
        train_cfg (dict, optional): 训练配置。默认为 None。
        test_cfg (list[dict], optional): 测试配置列表。默认为 [None, None]。
        with_pos_coord (bool): 是否将辅助头中的正样本作为额外的正查询。
            默认为 True。
        use_lsj (bool): 是否使用 Large Scale Jittering (LSJ) 进行数据增强。
            默认为 True。
        eval_module (str): 评估时使用的模块，可选 'detr', 'one-stage', 'two-stage'。
            默认为 'detr'。
        eval_index (int): 评估时使用的头部的索引。默认为 0。
        data_preprocessor (dict, optional): 数据预处理器的配置。默认为 None。
        init_cfg (dict, optional): 初始化配置。默认为 None。
    """

    def __init__(
            self,
            backbone,
            neck=None,
            query_head=None,  # detr head
            mask_roi_extractor=None,
            mask_head=None,
            mask_iou_head=None,     
            rpn_head=None,  # two-stage rpn
            roi_head=[None],  # two-stage
            bbox_head=[None],  # one-stage
            train_cfg: ConfigType = dict(),
            test_cfg=[None, None],
            # Control whether to consider positive samples
            # from the auxiliary head as additional positive queries.
            with_pos_coord=True,
            use_lsj=True,
            eval_module='detr',
            # Evaluate the Nth head.
            eval_index=0,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None):
        super(CoDETRInst, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.with_pos_coord = with_pos_coord
        self.use_lsj = use_lsj

        assert eval_module in ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        # 模块评估索引
        self.eval_index = eval_index

        # 初始化查询头
        if query_head is not None:
            query_head.update(train_cfg=train_cfg.query_head if (train_cfg and train_cfg.query_head is not None) else None)
            query_head.update(test_cfg=test_cfg.query_head)
            self.query_head = MODELS.build(query_head)
            self.query_head.init_weights()

        if mask_head is not None:
            """ 初始化 mask 头 """
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.mask_head = MODELS.build(mask_head)
            mask_train_cfg = train_cfg.mask_head if (train_cfg and train_cfg.mask_head is not None) else None
            self.rcnn_train_cfg = mask_train_cfg
            self.rcnn_test_cfg = test_cfg.mask_head
            if mask_train_cfg is not None:
                assigner = mask_train_cfg.assigner
                sampler = mask_train_cfg.sampler
                self.bbox_assigner = TASK_UTILS.build(assigner)
                self.bbox_sampler = TASK_UTILS.build(
                    sampler, default_args=dict(context=self))

        # mask_iou_head: 预测mask和真值之间的IoU
        if mask_iou_head is not None:
            """ 初始化 mask iou 头 """
            self.mask_iou_head = MODELS.build(mask_iou_head)

        if rpn_head is not None:
            """ 初始化 rpn 头 """
            rpn_train_cfg = train_cfg.two_stage_head[0].rpn if (
                train_cfg
                and train_cfg.two_stage_head is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.two_stage_head[0].rpn)
            self.rpn_head = MODELS.build(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                mask_train_cfg = train_cfg.two_stage_head[i].rcnn if (
                    train_cfg
                    and train_cfg.two_stage_head[i] is not None) else None
                roi_head[i].update(train_cfg=mask_train_cfg)
                roi_head[i].update(test_cfg=test_cfg.two_stage_head[i].rcnn)
                self.roi_head.append(MODELS.build(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(
                    train_cfg=train_cfg.bbox_head[i] if (
                        train_cfg and train_cfg.bbox_head is not None
                    ) else None)
                bbox_head[i].update(test_cfg=test_cfg.bbox_head[i])
                self.bbox_head.append(MODELS.build(bbox_head[i]))
                self.bbox_head[-1].init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: 检测器是否包含 RPN。"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: 检测器是否包含查询头。"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: 检测器是否包含 ROI 头。"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(
            self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: 检测器在 ROI 头中是否包含共享头。"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: 检测器是否包含 bbox 头。"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None
                 and len(self.roi_head) > 0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None
                    and len(self.bbox_head) > 0))

    @property
    def with_mask(self):
        """bool: 检测器是否包含 mask 头。"""
        return hasattr(self, 'mask_head') and self.mask_head is not None
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """提取特征。

        Args:
            batch_inputs (Tensor): 图像张量，形状为 (bs, dim, H, W)。

        Returns:
            tuple[Tensor]: 来自 neck 的特征图元组。每个特征图的形状为 (bs, dim, H, W)。
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        """在训练中运行 mask 头的正向函数并计算损失。

        Args:
            x (tuple[Tensor]): 多级图像特征的元组。
            sampling_results (list["obj:`SamplingResult`"]): 采样结果。
            batch_gt_instances (list[:obj:`InstanceData`]): 批量 gt_instance。
                通常包括 ``bboxes``, ``labels`` 和 ``masks`` 属性。

        Returns:
            dict: 通常返回一个字典，其中包含以下键：

                - `mask_preds` (Tensor): Mask 预测。
                - `loss_mask` (dict): Mask 损失组件的字典。
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))

        mask_loss_and_target = self.mask_head.loss_and_target(mask_results['stage_instance_preds_list'],
                                                   sampling_results,
                                                   batch_gt_instances,
                                                   rcnn_train_cfg=self.train_cfg.mask_head)
        mask_targets = mask_loss_and_target['mask_targets']
        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])

        if hasattr(self, "mask_iou_head"):
            # mask iou head forward and loss
            pos_mask_pred = mask_results['stage_instance_preds_list'][1].squeeze(1)
            mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                               pos_mask_pred)
            pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                            torch.cat(pos_labels)]
            loss_mask_iou = self.mask_iou_head.loss_and_target(
                pos_mask_iou_pred, 
                pos_mask_pred, 
                mask_targets[1], 
                sampling_results,
                batch_gt_instances, 
                self.train_cfg.mask_head)
            mask_results['loss_mask'].update(loss_mask_iou)
            mask_results.update(loss_mask_iou=loss_mask_iou)

        return mask_results

    def _mask_forward(self, x: Tuple[Tensor],
                      rois: Tensor,
                      roi_labels: Tensor) -> dict:
        """在训练和测试中使用的 Mask 头正向函数。

        Args:
            x (tuple[Tensor]): 多级图像特征的元组。
            rois (Tensor): 形状为 (n, 5) 的 ROI，其中第一列表示每个 ROI 的批次 ID。
            roi_labels (Tensor): ROI 的标签。

        Returns:
            dict: 通常返回一个字典，其中包含以下键：

                - `mask_preds` (Tensor): Mask 预测。
        """
        mask_roi_extractor = self.mask_roi_extractor
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)

        # SimpleRefineMaskHead
        stage_instance_preds_list, hidden_states = self.mask_head(mask_feats, x[0], rois, roi_labels)

        mask_results = dict(stage_instance_preds_list=stage_instance_preds_list, 
                            hidden_states=hidden_states, 
                            mask_feats=mask_feats)
        return mask_results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """计算损失。

        Args:
            batch_inputs (Tensor): 批量输入图像张量。
            batch_data_samples (SampleList): 批量数据样本。

        Returns:
            Union[dict, list]: 损失字典或列表。
        """
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        x = self.extract_feat(batch_inputs)

        losses = dict()

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # DETR 编码器和解码器正向传播
        if self.with_query_head:
            bbox_losses, x, results_list = self.query_head.loss(x, batch_data_samples)
            losses.update(bbox_losses)

        # mask head 正向传播和损失计算
        if hasattr(self, "mask_head"):
            num_imgs = len(batch_data_samples)
            outputs = unpack_gt_instances(batch_data_samples)
            batch_gt_instances, batch_gt_instances_ignore, _ = outputs

            sampling_results = []
            for i in range(num_imgs):
                results = results_list[i]  # InstanceData()
                results.priors = results.pop('bboxes')
                assign_result = self.bbox_assigner.assign(
                    results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            mask_results = self.mask_loss(x, sampling_results,
                                              batch_gt_instances)
            losses.update(mask_results['loss_mask'])
            if 'loss_mask_iou' in mask_results:
                losses.update(mask_results['loss_mask_iou'])

        # RPN 正向传播和损失计算
        if self.with_rpn:
            proposal_cfg = self.train_cfg.two_stage_head[0].get(
                'rpn_proposal', self.test_cfg.two_stage_head[0].rpn)

            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # 在 RPN 中将 gt_labels 的 cat_id 设置为 0
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            # 避免与 roi_head 损失同名
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # 使用 InstanceData 中预定义的 proposals 进行第二阶段的 ROI 特征提取。
            proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # roi 正向传播和损失计算
        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].loss(x, proposal_list,
                                               batch_data_samples)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    roi_losses.pop('pos_coords')
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)

        # CoATSSHead
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].loss(x, batch_data_samples)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    bbox_losses.pop('pos_coords')
            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.loss_aux(x, positive_coords[i],
                                                       i, batch_data_samples)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """使用后处理从一批输入和数据样本中预测结果。

        Args:
            batch_inputs (Tensor): 输入图像张量，形状为 (bs, dim, H, W)。
            batch_data_samples (List[:obj:`DetDataSample`]): 批量数据样本。
                通常包括诸如 `gt_instance` 或 `gt_panoptic_seg` 或 `gt_sem_seg` 等信息。
            rescale (bool): 是否重新缩放结果。默认为 True。

        Returns:
            list[:obj:`DetDataSample`]: 输入图像的检测结果。
            每个 DetDataSample 通常包含 'pred_instances'。并且
            `pred_instances` 通常包含以下键。

            - scores (Tensor): 分类分数，形状为 (num_instance, )
            - labels (Tensor): bbox 的标签，形状为 (num_instances, )。
            - bboxes (Tensor): 形状为 (num_instances, 4)，
              最后维度 4 排列为 (x1, y1, x2, y2)。
        """
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']

        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = img_metas['batch_input_shape']
                img_metas['img_shape'] = [input_img_h, input_img_w]

        img_feats = self.extract_feat(batch_inputs)
        if self.with_bbox and self.eval_module == 'one-stage':
            results_list = self.predict_bbox_head(
                img_feats, batch_data_samples, rescale=rescale)
        elif self.with_roi_head and self.eval_module == 'two-stage':
            results_list = self.predict_roi_head(
                img_feats, batch_data_samples, rescale=rescale)
        else:
            results_list = self.predict_query_head(
                img_feats, batch_data_samples, rescale=rescale)

            if hasattr(self, "mask_head"):
                results_list = self.predict_mask(
                    img_feats, batch_data_samples, results_list, rescale=rescale
                )

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """执行 mask 头的正向传播并预测上游网络特征上的检测结果。

        Args:
            x (tuple[Tensor]): 所有比例级别的特征图。
            batch_img_metas (list[dict]): 图像信息列表。
            results_list (list[:obj:`InstanceData`]): 每张图像的检测结果。
            rescale (bool): 如果为 True，则返回原始图像空间中的 bbox。默认为 False。

        Returns:
            list[:obj:`InstanceData`]: 后处理后每张图像的检测结果。
            每个项目通常包含以下键。

                - scores (Tensor): 分类分数，形状为 (num_instance, )
                - labels (Tensor): bbox 的标签，形状为 (num_instances, )。
                - bboxes (Tensor): 形状为 (num_instances, 4)，
                  最后维度 4 排列为 (x1, y1, x2, y2)。
                - masks (Tensor): 形状为 (num_instances, H, W)。
        """
        bboxes = [res.bboxes for res in results_list]
        labels = [res.labels for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_head.mask_thr_binary)
            return results_list
        
        mask_results = self._mask_forward(x, mask_rois, torch.cat(labels))
        # 只使用最后一阶段的 mask_preds
        mask_preds = mask_results['stage_instance_preds_list'][-1]
        # 将批量 mask 预测结果分割回每张图像
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg.mask_head,
            rescale=rescale)
        
        return results_list
    
    def predict_query_head(self,
                           mlvl_feats: Tuple[Tensor],
                           batch_data_samples: SampleList,
                           rescale: bool = True) -> InstanceList:
        """使用查询头进行预测。

        Args:
            mlvl_feats (tuple[Tensor]): 多级特征图元组。
            batch_data_samples (SampleList): 批量数据样本。
            rescale (bool): 是否重新缩放结果。默认为 True。

        Returns:
            InstanceList: 实例列表。
        """
        return self.query_head.predict(
            mlvl_feats, batch_data_samples=batch_data_samples, rescale=rescale)

    def predict_roi_head(self,
                         mlvl_feats: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         rescale: bool = True) -> InstanceList:
        """使用 ROI 头进行预测。

        Args:
            mlvl_feats (tuple[Tensor]): 多级特征图元组。
            batch_data_samples (SampleList): 批量数据样本。
            rescale (bool): 是否重新缩放结果。默认为 True。

        Returns:
            InstanceList: 实例列表。
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_query_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.query_head.forward(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        rpn_results_list = self.rpn_head.predict(
            mlvl_feats, batch_data_samples, rescale=False)
        return self.roi_head[self.eval_index].predict(
            mlvl_feats, rpn_results_list, batch_data_samples, rescale=rescale)

    def predict_bbox_head(self,
                          mlvl_feats: Tuple[Tensor],
                          batch_data_samples: SampleList,
                          rescale: bool = True) -> InstanceList:
        """使用 bbox 头进行预测。

        Args:
            mlvl_feats (tuple[Tensor]): 多级特征图元组。
            batch_data_samples (SampleList): 批量数据样本。
            rescale (bool): 是否重新缩放结果。默认为 True。

        Returns:
            InstanceList: 实例列表。
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_query_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.query_head.forward(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        return self.bbox_head[self.eval_index].predict(
            mlvl_feats, batch_data_samples, rescale=rescale)
