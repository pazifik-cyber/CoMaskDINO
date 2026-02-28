# Copyright (c) OpenMMLab. All rights reserved.
"""
CoMaskDINO: 在 MaskDINO 基础上引入 Co-DETR 协同训练机制。

核心思想：
  主预测路径  : MaskDINO (panoptic_head → panoptic_fusion_head)
                提供 mask + box + class 联合预测损失。
  辅助预测路径: RPN + RoI Head (两阶段) + CoATSS Head (一阶段)
                提供额外的目标检测监督。
  协同反馈    : 辅助头产生的高质量正样本坐标 (pos_coords) 被反馈给
                CoMaskDINOHead.loss_aux()，作为额外正查询驱动主解码器
                学习更好的表示。
"""

import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models import MaskFormer
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class CoMaskDINO(MaskFormer):
    """CoMaskDINO: Collaborative Training MaskDINO with Auxiliary Detection Heads.

    在标准 MaskDINO 的基础上，引入 Co-DETR 的协同训练框架：
    - panoptic_head    : CoMaskDINOHead，负责主要的 mask/box/class 预测。
    - rpn_head         : 区域候选网络，提供两阶段训练的 proposal。
    - roi_head         : RoI 头列表，提供高质量 bbox 正样本 (pos_coords)。
    - bbox_head        : 一阶段 CoATSS 头列表，提供锚框正样本 (pos_coords)。

    Args:
        backbone (dict):               主干网络配置。
        neck (dict, optional):         颈部网络配置 (FPN)。
        panoptic_head (dict):          CoMaskDINOHead 配置（主预测头）。
        panoptic_fusion_head (dict):   MaskDINOFusionHead 配置（推理后处理）。
        rpn_head (dict, optional):     RPNHead 配置，用于生成 proposal。
        roi_head (list[dict]):         CoStandardRoIHead 配置列表。
        bbox_head (list[dict]):        CoATSSHead 配置列表。
        train_cfg (dict):              训练配置（同时包含 MaskDINO 和辅助头参数）。
        test_cfg (dict):               测试配置。
        with_pos_coord (bool):         是否将辅助头正样本坐标反馈给主头。默认 True。
        use_lsj (bool):                是否使用 LSJ 数据增强修正 img_shape。默认 True。
        data_preprocessor (dict):      数据预处理器配置。
        init_cfg (dict):               参数初始化配置。
    """

    def __init__(
            self,
            backbone: ConfigType,
            neck: OptConfigType = None,
            panoptic_head: OptConfigType = None,
            panoptic_fusion_head: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: list = [None],
            bbox_head: list = [None],
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            with_pos_coord: bool = True,
            use_lsj: bool = True,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None):

        # MaskFormer.__init__ 会构建 backbone/neck/panoptic_head/panoptic_fusion_head，
        # 并将 train_cfg / test_cfg 传递给 panoptic_head。
        # CoMaskDINOHead.__init__ 内部会过滤掉辅助头相关的 cfg 键。
        super().__init__(
            backbone=backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.with_pos_coord = with_pos_coord
        self.use_lsj = use_lsj

        # ------------------------------------------------------------------ #
        # 构建辅助预测头
        # ------------------------------------------------------------------ #
        # train_cfg 结构（来自 config）：
        #   rpn         : RPN 训练配置
        #   rpn_proposal: RPN proposal 后处理配置
        #   rcnn        : list[dict], RoI 头训练配置
        #   bbox_head   : list[dict], ATSS 头训练配置
        # test_cfg 结构：
        #   rpn         : RPN 测试配置
        #   rcnn        : list[dict], RoI 头测试配置
        #   bbox_head   : list[dict], ATSS 头测试配置
        # ------------------------------------------------------------------ #

        # ---------- RPN Head ----------
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.get('rpn', None) if train_cfg else None
            rpn_test_cfg = test_cfg.get('rpn', None) if test_cfg else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=rpn_test_cfg)
            self.rpn_head = MODELS.build(rpn_head_)
            self.rpn_head.init_weights()

        # ---------- RoI Heads ----------
        rcnn_train_cfgs = train_cfg.get('rcnn', []) if train_cfg else []
        rcnn_test_cfgs = test_cfg.get('rcnn', []) if test_cfg else []
        self.roi_head = nn.ModuleList()
        for i, rh_cfg in enumerate(roi_head):
            if rh_cfg is not None:
                rh = rh_cfg.copy()
                rh.update(train_cfg=rcnn_train_cfgs[i]
                           if i < len(rcnn_train_cfgs) else None)
                rh.update(test_cfg=rcnn_test_cfgs[i]
                           if i < len(rcnn_test_cfgs) else None)
                self.roi_head.append(MODELS.build(rh))
                self.roi_head[-1].init_weights()

        # ---------- Bbox Heads (CoATSS) ----------
        bh_train_cfgs = train_cfg.get('bbox_head', []) if train_cfg else []
        bh_test_cfgs = test_cfg.get('bbox_head', []) if test_cfg else []
        self.bbox_head = nn.ModuleList()
        for i, bh_cfg in enumerate(bbox_head):
            if bh_cfg is not None:
                bh = bh_cfg.copy()
                bh.update(train_cfg=bh_train_cfgs[i]
                           if i < len(bh_train_cfgs) else None)
                bh.update(test_cfg=bh_test_cfgs[i]
                           if i < len(bh_test_cfgs) else None)
                self.bbox_head.append(MODELS.build(bh))
                self.bbox_head[-1].init_weights()

    # ------------------------------------------------------------------ #
    # 属性访问
    # ------------------------------------------------------------------ #

    @property
    def with_rpn(self) -> bool:
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        return (hasattr(self, 'roi_head') and
                self.roi_head is not None and
                len(self.roi_head) > 0)

    @property
    def with_bbox_head(self) -> bool:
        return (hasattr(self, 'bbox_head') and
                self.bbox_head is not None and
                len(self.bbox_head) > 0)

    # ------------------------------------------------------------------ #
    # 前向传播 (必须实现的抽象方法)
    # ------------------------------------------------------------------ #

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    # ------------------------------------------------------------------ #
    # 训练：loss
    # ------------------------------------------------------------------ #

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """计算所有损失。

        训练流程：
        1. extract_feat  → FPN 特征 x
        2. panoptic_head.loss  → MaskDINO 主损失 (cls + box + mask)
        3. rpn_head  → proposal_list + rpn_losses
        4. roi_head[i].loss  → roi_losses + pos_coords_roi
        5. bbox_head[i].loss → atss_losses + pos_coords_atss
        6. panoptic_head.loss_aux(pos_coords_*)  → aux 协同损失

        Returns:
            dict: 所有损失的字典，键值唯一不冲突。
        """
        # --- LSJ 模式下修正 img_shape ---
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for ds in batch_data_samples:
                h, w = batch_input_shape
                ds.metainfo['img_shape'] = [h, w]

        # 1. 提取多尺度特征
        x = self.extract_feat(batch_inputs)

        losses = dict()

        def _upd_loss(raw_losses: dict, idx: int, weight: float = 1.0) -> dict:
            """给损失键追加下标，避免多个辅助头的损失键冲突。"""
            return {
                f'{k}{idx}': ([v_ * weight for v_ in v]
                              if isinstance(v, (list, tuple))
                              else v * weight)
                for k, v in raw_losses.items()
            }

        # 2. MaskDINO 主损失
        maskdino_losses = self.panoptic_head.loss(x, batch_data_samples)
        losses.update(maskdino_losses)

        # 3. RPN 前向 + 损失
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal',
                self.test_cfg.get('rpn', None))
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # RPN 只区分前景/背景，将所有类别 label 改为 0
            for ds in rpn_data_samples:
                ds.gt_instances.labels = torch.zeros_like(
                    ds.gt_instances.labels)
            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # 避免与 roi_head 损失同名
            for k in list(rpn_losses.keys()):
                if 'loss' in k and 'rpn' not in k:
                    rpn_losses[f'rpn_{k}'] = rpn_losses.pop(k)
            losses.update(rpn_losses)
        else:
            # 使用预先提供的 proposal
            proposal_list = [ds.proposals for ds in batch_data_samples]

        # 4. RoI Head：两阶段 bbox 预测，并收集正样本坐标
        positive_coords = []
        for i, roi_head in enumerate(self.roi_head):
            roi_losses = roi_head.loss(x, proposal_list, batch_data_samples)
            if self.with_pos_coord and 'pos_coords' in roi_losses:
                positive_coords.append(roi_losses.pop('pos_coords'))
            elif 'pos_coords' in roi_losses:
                roi_losses.pop('pos_coords')
            losses.update(_upd_loss(roi_losses, idx=i))

        # 5. CoATSS Head：一阶段 bbox 预测，并收集正样本坐标
        for i, bbox_head in enumerate(self.bbox_head):
            bbox_losses = bbox_head.loss(x, batch_data_samples)
            if self.with_pos_coord and 'pos_coords' in bbox_losses:
                positive_coords.append(bbox_losses.pop('pos_coords'))
            elif 'pos_coords' in bbox_losses:
                bbox_losses.pop('pos_coords')
            losses.update(_upd_loss(bbox_losses, idx=i + len(self.roi_head)))

        # 6. 协同辅助损失：将正样本坐标反馈给主头的解码器
        if self.with_pos_coord and len(positive_coords) > 0:
            for i, pos_coords in enumerate(positive_coords):
                aux_losses = self.panoptic_head.loss_aux(
                    x, pos_coords, i, batch_data_samples)
                losses.update(_upd_loss(aux_losses, idx=i))

        return losses

    # ------------------------------------------------------------------ #
    # 推理：predict
    # ------------------------------------------------------------------ #

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """推理阶段仅使用主路径（MaskDINO），不需要辅助头。

        Args:
            batch_inputs (Tensor): 批量输入图像，形状 (bs, C, H, W)。
            batch_data_samples (SampleList): 批量数据样本。
            rescale (bool): 是否将预测结果缩放回原始图像尺度。

        Returns:
            SampleList: 每张图像的预测结果（包含 pred_instances）。
        """
        feats = self.extract_feat(batch_inputs)
        mask_cls_results, mask_pred_results, mask_box_results = \
            self.panoptic_head.predict(feats, batch_data_samples)
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            mask_box_results,
            batch_data_samples,
            rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)
