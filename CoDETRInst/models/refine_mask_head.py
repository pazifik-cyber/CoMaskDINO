# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright 2024 [Your Name/Organization]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair 

import numpy as np
from typing import List, Tuple, Optional

from mmcv.cnn import ConvModule, build_upsample_layer # MMCV中的卷积模块和上采样层构建器
from mmcv.ops import RoIAlign  # MMCV中的RoIAlign操作

from mmengine.model import BaseModule, ModuleList # 模型和模块列表的基类
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData # 用于存储单个图像实例信息的结构
from torch import Tensor 

from mmdet.registry import MODELS # 用于注册模型和损失函数的注册器
from mmdet.structures.mask import mask_target, BitmapMasks, PolygonMasks # Mask相关结构和目标生成函数
from mmdet.models.utils import empty_instances # 创建空实例数据的工具函数
from mmdet.models.task_modules.samplers import SamplingResult # 采样结果的数据结构
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig # MMDetection中常用的类型别名
from mmengine.config import Config

BYTES_PER_FLOAT = 4 # 每个浮点数占用的字节数
GPU_MEM_LIMIT = 1024**3  # 1 GB 显存限制 (用于推理时mask粘贴操作的分块)

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask


class MultiBranchFusion(BaseModule):
    """多分支融合模块 (Multi-branch fusion module)。

    使用不同空洞率的卷积分支进行特征融合。
    """
    def __init__(self,
                 feat_dim: int, # 输入/输出特征维度
                 dilations: List[int] = [1, 3, 5], # 空洞卷积的空洞率列表
                 conv_cfg: OptConfigType = None, # 卷积层配置
                 norm_cfg: OptConfigType = None, # 归一化层配置
                 init_cfg: OptMultiConfig = None): # 初始化配置
        super().__init__(init_cfg=init_cfg)
        self.dilation_convs = ModuleList() # 用于存储不同空洞率的卷积层
        for idx, dilation in enumerate(dilations):
            self.dilation_convs.append(ConvModule(
                feat_dim,
                feat_dim,
                kernel_size=3,
                padding=dilation, # 根据空洞率设置padding以保持分辨率
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        # 用于合并多分支特征的1x1卷积层
        self.merge_conv = ConvModule(
            feat_dim,
            feat_dim,
            kernel_size=1,
            act_cfg=None, # 通常在融合块之后应用ReLU激活
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg) # 这里有时也可以使用Norm层

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。"""
        # 计算每个空洞卷积分支的特征
        feats = [conv(x) for conv in self.dilation_convs]
        # 将各分支特征相加，并通过1x1卷积进行融合
        out_feat = self.merge_conv(sum(feats))
        return out_feat

class MultiBranchFusionAvg(MultiBranchFusion):
    """带平均池化分支的多分支融合模块。

    在MultiBranchFusion的基础上增加了一个全局平均池化分支。
    """
    def forward(self, x: Tensor) -> Tensor:
        """前向传播。"""
        # 计算空洞卷积分支的特征
        dilation_feats = [conv(x) for conv in self.dilation_convs]
        # 计算全局平均池化特征
        avg_feat = F.avg_pool2d(x, x.shape[-2:]) # 对最后两个维度(H, W)进行全局平均池化
        # 可选：如果merge_conv期望输入具有相同的空间维度，则需要扩展avg_feat
        # avg_feat = avg_feat.expand_as(dilation_feats[0])
        # 将所有分支（空洞卷积 + 平均池化）的特征相加，并通过1x1卷积融合
        out_feat = self.merge_conv(sum(dilation_feats) + avg_feat)
        return out_feat


class SimpleSFMStage(BaseModule):
    """新版的语义融合掩码优化阶段 (Semantic Fusion Mask Stage)。
    """

    def __init__(self,
                 semantic_in_channel: int = 256,
                 semantic_out_channel: int = 256,
                 instance_in_channel: int = 256,
                 instance_out_channel: int = 256,
                 fusion_type: str = 'MultiBranchFusionAvg',
                 dilations: List[int] = [1, 3, 5],
                 out_size: int = 14,
                 semantic_out_stride: int = 4,
                 upsample_cfg: ConfigType = dict(
                     type='bilinear', scale_factor=2),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        self.semantic_transform_in = nn.Conv2d(semantic_in_channel,
                                               semantic_out_channel, 1)

        # 使用 MODELS.build 构建 RoI 提取器
        self.semantic_roi_extractor = MODELS.build(
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=out_size, sampling_ratio=0),
                out_channels=semantic_out_channel,
                featmap_strides=[semantic_out_stride]))

        fuse_in_channel = instance_in_channel + semantic_out_channel + 1

        # 实例化融合模块
        fusion_module = MultiBranchFusionAvg(
            instance_in_channel, dilations=dilations)

        self.fuse_conv = ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1), fusion_module
        ])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel,
                                            instance_out_channel - 1, 1)
        # 使用 mmengine.model.build_upsample_layer 构建上采样层
        self.upsample = build_upsample_layer(upsample_cfg)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        """初始化权重。"""
        super().init_weights()
        for m in [self.semantic_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats: Tensor, instance_logits: Tensor,
                semantic_feat: Tensor, rois: Tensor,
                upsample: bool = True) -> Tuple[Tensor, Tensor]:
        """前向传播。"""
        # 提取实例级别的语义特征
        semantic_feat_transformed = self.relu(
            self.semantic_transform_in(semantic_feat))
        # RoI 提取器期望特征图是一个列表或元组
        ins_semantic_feats = self.semantic_roi_extractor(
            (semantic_feat_transformed, ), rois)

        # 融合实例特征、实例掩码和语义特征
        concat_tensors = [
            instance_feats, ins_semantic_feats,
            instance_logits.sigmoid()
        ]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        # 再次与实例掩码拼接
        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        # 在上采样前拼接，与论文保持一致
        fused_feats = torch.cat([fused_feats,
                                 instance_logits.sigmoid()],
                                dim=1)
        if upsample:
            fused_feats = self.upsample(fused_feats)
        return fused_feats, semantic_feat_transformed



@MODELS.register_module()
class SimpleRefineMaskHead(BaseModule):
    """简化的RefineMask头，使用语义流模块 (SFM)。

    该Mask Head通过多个阶段逐步优化Mask预测结果。
    """
    def __init__(self,
                 num_convs_instance: int = 2, # 初始实例分支卷积层数
                 num_convs_semantic: int = 4, # 初始语义分支卷积层数
                 conv_in_channels_instance: int = 256, # 实例分支输入通道数 (通常来自RoI Extractor)
                 conv_in_channels_semantic: int = 256, # 语义分支输入通道数 (通常来自Neck)
                 conv_kernel_size_instance: int = 3, # 实例分支卷积核大小
                 conv_kernel_size_semantic: int = 3, # 语义分支卷积核大小
                 conv_out_channels_instance: int = 256, # 实例分支输出通道数
                 conv_out_channels_semantic: int = 256, # 语义分支输出通道数
                 num_stages: int = 3, # Refinement阶段数量 (例如, 3个阶段意味着共4次预测)
                 stage_channels_factor: int = 2, # 每个阶段通道数减少因子 (例如, 2表示通道数减半)
                 # 每个阶段开始时用于监督的RoI特征图目标尺寸 (列表长度等于num_stages)
                 stage_sup_size: List[int] = [14, 28, 56, 112],
                 semantic_out_stride: int = 4, # 输入语义特征图相对于原图的步长
                 fusion_type: str = 'MultiBranchFusionAvg', # SFM阶段使用的融合模块类型
                 dilations: List[int] = [1, 3, 5], # SFM阶段融合模块的空洞率
                 # 是否在最后一个阶段的特征计算logits之前进行上采样
                 pre_upsample_last_stage: bool = False,
                 # 上采样层配置
                 upsample_cfg: ConfigType = dict(type='bilinear', scale_factor=2, align_corners=False),
                 conv_cfg: OptConfigType = None, # 卷积层配置
                 norm_cfg: OptConfigType = dict(type='GN', num_groups=32, requires_grad=True), # 归一化层配置 (例如GroupNorm)
                 loss_cfg: ConfigType = dict(
                    type='BARCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1), 
                 loss_weight=1.0,
                 stage_num_classes: List[int] = [80, 80, 80, 80], 
                 init_cfg: OptMultiConfig = None): # 初始化配置
        assert init_cfg is None, '为了防止异常的初始化行为, 不允许设置init_cfg'
        super().__init__(init_cfg=init_cfg)

        # --- 保存配置参数 ---
        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.num_stages = num_stages
        self.stage_channels_factor = stage_channels_factor
        self.stage_sup_size = stage_sup_size

        self.stage_num_classes = stage_num_classes 
        if len(self.stage_num_classes) != num_stages + 1: # <--- 添加长度检查
            raise ValueError(
                f"`stage_num_classes` (长度={len(self.stage_num_classes)}) "
                f"必须等于 `num_stages` + 1 ({num_stages + 1})")

        self.semantic_out_stride = semantic_out_stride
        self.pre_upsample_last_stage = pre_upsample_last_stage
        self.upsample_cfg = upsample_cfg.copy()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.class_agnostic = True if self.stage_num_classes[-1] == 1 else False

        # --- 损失函数配置 ---
        # 构建基础的Mask损失函数
        self.loss_mask = MODELS.build(loss_cfg)
        self.loss_weight = loss_weight
        # --- 构建网络层 ---
        self._build_conv_layer('instance') # 构建实例分支的初始卷积层
        self._build_conv_layer('semantic') # 构建语义分支的初始卷积层

        self.stages = ModuleList() # 用于存储SFM阶段模块
        self.stage_instance_logits = ModuleList() # 用于存储每个阶段输出Logits的卷积层

        # --- 构建Refinement阶段 ---
        instance_channel = self.conv_out_channels_instance # 初始实例特征通道数
        semantic_channel = self.conv_out_channels_semantic # 初始语义特征通道数
        stage_out_channels = [instance_channel] # 记录每个阶段 *输入* 的实例特征通道数

        for i in range(self.num_stages):
            # 计算下一个阶段的实例特征通道数
            next_instance_channel = instance_channel // self.stage_channels_factor

            # 创建并添加一个SFM阶段
            stage = SimpleSFMStage(
                semantic_in_channel=semantic_channel, # 语义分支输入通道
                semantic_out_channel=instance_channel, # 语义分支输出通道 (与实例分支融合前匹配)
                instance_in_channel=instance_channel, # 实例分支输入通道
                instance_out_channel=next_instance_channel, # 实例分支输出通道 (下一阶段输入)
                fusion_type=fusion_type, # 融合类型
                dilations=dilations, # 空洞率
                # SFM内部RoIAlign的目标尺寸使用当前阶段的监督尺寸
                out_size=self.stage_sup_size[i],
                semantic_out_stride=self.semantic_out_stride, # 语义特征步长
                upsample_cfg=self.upsample_cfg, # 上采样配置
            )
            self.stages.append(stage)
            # 更新当前实例特征通道数，为下一个循环做准备
            instance_channel = next_instance_channel
            stage_out_channels.append(instance_channel) # 记录下一阶段的输入通道数

        self.stage_instance_logits = nn.ModuleList([
                nn.Conv2d(stage_out_channels[idx], num_classes, 1) for idx, num_classes in enumerate(self.stage_num_classes)])

        self.relu = nn.ReLU(inplace=True) # ReLU激活

    def _build_conv_layer(self, name: str) -> None:
        """构建实例或语义分支的初始卷积层。"""
        convs = ModuleList() # 存储该分支的卷积层
        # 获取该分支的配置参数
        num_convs = getattr(self, f'num_convs_{name}')
        in_channels = getattr(self, f'conv_in_channels_{name}')
        out_channels = getattr(self, f'conv_out_channels_{name}')
        kernel_size = getattr(self, f'conv_kernel_size_{name}')
        cur_channels = in_channels # 当前层的输入通道数
        for i in range(num_convs):
            convs.append(ConvModule(
                cur_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2, # 标准padding保持分辨率
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
                ))
            cur_channels = out_channels
        self.add_module(f'{name}_convs', convs)

    def init_weights(self) -> None:
        """初始化权重。"""
        super().init_weights() # 调用父类的初始化 (会初始化ConvModule等)
        # 初始化所有Logits预测层的权重和偏置
        for logits_layer in self.stage_instance_logits:
            nn.init.kaiming_normal_(logits_layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(logits_layer.bias, 0)
        # SFM Stage是BaseModule，它们会自我初始化


    def forward(self,
                instance_feats: Tensor,
                semantic_feat: Tensor,
                rois: Tensor,
                roi_pred_classes: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """

        此函数负责执行多阶段的掩码优化过程。它接收实例级别的特征、
        语义级别的特征以及每个提议框的预测类别，并逐阶段地输出
        更精细的掩码预测。

        参数:
            instance_feats (Tensor): 从 RoI Align 或其他池化操作获得的实例特征。
                形状为 [num_rois, C, H, W]。
            semantic_feat (Tensor): 从 FPN 获取的语义特征。
            rois (Tensor): RoI 提议框，形状为 [num_rois, 4] 或 [num_rois, 5]。
                在您的旧代码中，此参数被传递给了 stage，因此予以保留。
            roi_pred_classes (Tensor): 每个 RoI 预测的类别索引。
                形状为 [num_rois]。

        返回:
            Tuple[List[Tensor], List[Tensor]]:
                - stage_instance_preds (List[Tensor]): 每个优化阶段的实例分割预测。
                - hidden_states (List[Tensor]): 用于调试或辅助损失的所有中间特征状态。
        """
        # 记录中间特征用于可能的辅助损失
        hidden_states = []

        # --- 1. 初始卷积层 ---
        # instance_feats 来自 RoI Extractor
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        # semantic_feat 来自 FPN
        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        hidden_states.append(instance_feats)
        hidden_states.append(semantic_feat)

        stage_instance_preds = []
        num_rois = instance_feats.size(0)

        # --- 2. 逐阶段进行特征优化 ---
        for idx, stage in enumerate(self.stages):  # 3个stage
            # 从当前阶段的logits预测模块获取所有类别的logits
            # 形状: [num_rois, num_classes, h, w]
            all_class_logits = self.stage_instance_logits[idx](instance_feats)

            # 根据每个RoI预测的类别，提取对应的logits
            # 形状: [num_rois, 1, h, w]
            instance_logits = all_class_logits[torch.arange(num_rois),
                                               roi_pred_classes][:, None]
            stage_instance_preds.append(instance_logits)

            # 决定是否需要在stage内部进行上采样
            # 最后一个stage的上采样在外部处理（如果需要）
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1

            # 将特征和初步预测传入SFM stage进行融合与优化
            # 注意：根据您的旧代码，rois 也被传入 stage，这里予以保留
            instance_feats, tmp_semantic_feats = stage(instance_feats,
                                                       instance_logits,
                                                       semantic_feat,
                                                       rois,
                                                       upsample_flag)

            hidden_states.append(instance_feats)
            hidden_states.append(tmp_semantic_feats)

        # --- 3. 处理最后一个阶段的预测 ---
        if self.stage_num_classes[-1] == 1:
            # 将所有预测类别强制钳位到索引 0
            final_roi_pred_classes = roi_pred_classes.clamp(max=0)
        else:
            final_roi_pred_classes = roi_pred_classes

        # 获取最后一个阶段的最终预测
        # 形状: [num_rois, 1, h, w]
        instance_preds = self.stage_instance_logits[-1](
            instance_feats)[torch.arange(num_rois), final_roi_pred_classes][:,
                                                                            None]

        # 根据配置，在预测后进行上采样（这是更常见的做法）
        if not self.pre_upsample_last_stage:
            instance_preds = F.interpolate(
                instance_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=True)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds, hidden_states


    def get_targets(self,
                    sampling_results: List['SamplingResult'],
                    batch_gt_instances: 'InstanceList',
                    rcnn_train_cfg: 'ConfigDict') -> List[Tensor]:
        """为每个阶段生成掩码目标。

        该函数使目标生成过程适配 MMDetection 3.x 的 API，使用了
        `SamplingResult` 和 `InstanceData` 对象。它为每个优化阶段调用
        标准的 `mask_target` 工具函数，以生成不同分辨率的目标。

        参数:
            sampling_results (List[SamplingResult]): 一个批次中每张图像的
                采样结果。
            batch_gt_instances (InstanceList): 一个批次中每张图像的
                真实实例。包含 bboxes、labels 和 masks。
            rcnn_train_cfg (ConfigDict): R-CNN 头的训练配置。

        返回:
            List[Tensor]: 每个优化阶段的掩码目标列表。每个张量包含了该
            特定阶段批次中所有正样本提案的拼接目标。
        """
        # 提取正样本提案及其对应的真实标签信息
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [gt_inst.masks for gt_inst in batch_gt_instances]

        # 为每个监督阶段生成目标
        stage_instance_targets = []
        # 创建 train_cfg 的副本以避免修改原始配置
        train_cfg = rcnn_train_cfg.copy()

        for mask_size in self.stage_sup_size:
            # 为当前阶段设置目标掩码大小
            train_cfg.mask_size = _pair(mask_size)
            targets_for_stage = mask_target(pos_proposals,
                                            pos_assigned_gt_inds, gt_masks,
                                            train_cfg)
            stage_instance_targets.append(targets_for_stage)

        return stage_instance_targets


    def loss_and_target(self,
                        stage_instance_preds: List[Tensor],
                        sampling_results: List['SamplingResult'],
                        batch_gt_instances: 'InstanceList',
                        rcnn_train_cfg: 'ConfigDict') -> dict:
        """根据头部提取的特征计算损失并生成目标。

        参数:
            stage_instance_preds (List[Tensor]): 模型前向传播输出的每个优化
                阶段的预测掩码，列表中的每个张量形状为 (num_pos, num_classes, h, w)。
            sampling_results (List[SamplingResult]): 一个批次中每张图像的
                采样结果。
            batch_gt_instances (InstanceList): 一个批次中每张图像的
                真实实例。包含 bboxes、labels 和 masks。
            rcnn_train_cfg (ConfigDict): R-CNN 头的训练配置。

        返回:
            dict: 一个包含损失和目标组件的字典。
        """
        # 为每个阶段获取真实的掩码目标
        stage_instance_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        losses = dict()
        # 检查是否存在正样本
        if stage_instance_preds[0].size(0) == 0:
            # 如果没有正样本，返回一个零损失，同时确保所有相关参数都能接收到梯度
            # 这对于分布式数据并行（DDP）模式很重要
            loss_instance = 0.
            for pred in stage_instance_preds:
                loss_instance += pred.sum() * 0
            losses['loss_mask'] = loss_instance
        else:
            # 调用自定义的多阶段损失函数进行计算
            loss_instance = self.loss_mask(stage_instance_preds,
                                           stage_instance_targets) * self.loss_weight
            losses['loss_mask'] = loss_instance

        return dict(loss_mask=losses, mask_targets=stage_instance_targets)


    def predict_by_feat(self,
                        mask_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False) -> InstanceList:
        """根据特征图转换掩码结果

        参数:
            mask_preds (Tuple[Tensor]):  Tuple of predicted foreground masks,
                each has shape (n, num_classes, h, w)
            results_list (List[InstanceData]): 每张图像的检测结果。
            batch_img_metas (List[dict]): 每张图像的元信息。
            rcnn_test_cfg (ConfigDict): R-CNN 的测试配置。
            rescale (bool): 是否将坐标缩放回原始图像尺寸。

        返回:
            InstanceList: 更新了 `.masks` 属性的检测结果列表。
        """
        assert len(mask_preds) == len(results_list)
        for img_id in range(len(results_list)):
            results = results_list[img_id]
            bboxes = results.bboxes
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary)[0]
            else:
                img_meta = batch_img_metas[img_id]
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale)
                results.masks = im_mask

        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        """从 mask_preds 和 bboxes 获取分割掩码。

        Args:
            mask_preds (Tensor): 预测的前景掩码，形状为
                (n, num_classes, h, w)。
            bboxes (Tensor): 预测的边界框，形状为 (n, 4)
            labels (Tensor): 边界框的标签，形状为 (n, )
            img_meta (dict): 图像信息。
            rcnn_test_cfg (obj:`ConfigDict`): Bbox Head 的 `test_cfg`。
                默认为 None。
            rescale (bool): 如果为 True，则在原始图像空间中返回框。
                默认为 False。
            activate_map (bool): 是否通过增强测试获得结果。
                如果为 True，`mask_preds` 将不使用 sigmoid 处理。
                默认为 False。

        Returns:
            Tensor: 编码后的掩码，形状为 (n, img_w, img_h)

        Example:
            >>> from mmengine.config import Config
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_preds = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = (1, 1)
            >>> rescale = False
            >>> img_meta = {'scale_factor': scale_factor,
            ...             'ori_shape': ori_shape}
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self._get_seg_masks_single(
            ...     mask_preds, bboxes, labels,
            ...     img_meta, rcnn_test_cfg, rescale)
            >>> assert encoded_masks.size()[0] == N
            >>> assert encoded_masks.size()[1:] == ori_shape
        """
        # 从 img_meta 中获取缩放因子
        scale_factor = bboxes.new_tensor(img_meta.scale_factor).repeat(
            (1, 2))
        # 从 img_meta 中获取原始图像的高度和宽度
        img_h, img_w = img_meta.ori_shape[:2]
        # 获取 bboxes 所在的设备
        device = bboxes.device

        # 如果不是在增强测试模式下
        if not activate_map:
            # 对掩码预测结果应用 sigmoid 函数以获得概率
            mask_preds = mask_preds.sigmoid()
        else:
            # 在增强测试（AugTest）中，掩码在之前已经激活
            # 直接将 mask_preds 转换为张量
            mask_preds = bboxes.new_tensor(mask_preds)

        # 如果需要将边界框缩放回原始图像尺寸
        if rescale:  # 原地缩放边界框
            bboxes /= scale_factor
        else:
            # 如果不缩放，则根据缩放因子调整图像的尺寸
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        # 获取掩码预测的数量
        N = len(mask_preds)
        # 实际实现将输入分成块，并逐块粘贴。
        if device.type == 'cpu':
            # 在 CPU 上，当逐个粘贴并设置 skip_empty=True 时效率最高，
            # 这样可以执行最少的操作。
            num_chunks = N
        else:
            # GPU 从较大块的并行处理中受益，但可能会有内存问题
            # img_w 和 img_h 的类型是 np.int32,
            # 当图像分辨率很大时,
            # num_chunks 的计算会溢出。
            # 所以我们需要将 img_w 和 img_h 的类型改为 int。
            # 参见 https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            # 断言块数不大于 N，否则意味着 GPU_MEM_LIMIT 太小
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        # 将所有 proposal 的索引分成多个块
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        # 从测试配置中获取二值化阈值
        threshold = rcnn_test_cfg.mask_thr_binary
        # 创建一个全零张量来存储最终的图像掩码
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            # 如果阈值大于等于0，则使用布尔类型，否则使用 uint8
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        # 如果不是类别无关的（class-agnostic）
        if not self.class_agnostic:
            # 根据每个 proposal 的标签选择对应的类别掩码
            mask_preds = mask_preds[range(N), labels][:, None]

        # 遍历每个块
        for inds in chunks:
            # 调用 _do_paste_mask 函数将块中的掩码粘贴到正确的位置
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_preds[inds],  # 当前块的掩码预测
                bboxes[inds],  # 当前块的边界框
                img_h,  # 图像高度
                img_w,  # 图像宽度
                skip_empty=device.type == 'cpu')  # 在 CPU 上跳过空区域以提高效率

            # 如果设置了二值化阈值
            if threshold >= 0:
                # 将掩码块二值化
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # 用于可视化和调试，将掩码值缩放到 0-255 范围
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            # 将处理后的掩码块粘贴到最终的图像掩码张量中
            im_mask[(inds, ) + spatial_inds] = masks_chunk
        # 返回最终的实例掩码
        return im_mask


def _do_paste_mask(masks: Tensor,
                   boxes: Tensor,
                   img_h: int,
                   img_w: int,
                   skip_empty: bool = True) -> tuple:
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
        is the slice object.

            If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.

            If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
