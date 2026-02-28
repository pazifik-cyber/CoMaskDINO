# Copyright (c) OpenMMLab. All rights reserved.
"""
CoMaskDINOHead: MaskDINOHead + Co-DETR 协同训练辅助损失。

设计原则（从根本目标出发）：
  目标是提升分割精度，不只是检测。
  Co-DETR 机制的价值在于：辅助头（RPN/ROI/ATSS）提供稠密正样本坐标
  → 注入 MaskDINO 解码器作为额外查询
  → 这些查询同时接受 cls + box + mask 监督
  → 解码器得到比主路径（稀疏 Hungarian 匹配）多一个数量级的 mask 监督信号
  → encoder 特征图、decoder attention 均变得更具判别性

  因此，aux 路径必须完整走 mask_embed + pixel decoder，不能只做 cls+box。

模块结构：
  loss_aux()               ← CoMaskDINO.loss() 调用入口
    ├─ pixel_decoder        共享权重，得到 mask_features + multi_scale_feats
    ├─ get_aux_targets()    pos_coords → 查询初始化 + GT mask 匹配
    ├─ forward_aux_decoder()     → cls + box + mask 三路预测
    └─ loss_aux_by_feat()   → cls/bbox/iou/mask/dice 全损失
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.losses import GIoULoss, L1Loss
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import reduce_mean

from projects.MaskDINO.maskdino import MaskDINOHead

# SetCriterion 不接受的 train_cfg 键（属于辅助头配置）
_AUX_CFG_KEYS = frozenset(['rpn', 'rpn_proposal', 'rcnn', 'bbox_head'])


@MODELS.register_module()
class CoMaskDINOHead(MaskDINOHead):
    """CoMaskDINO 的主预测头。

    在 MaskDINOHead 基础上新增协同训练辅助损失路径。
    关键改进：辅助正样本查询走完整的 cls+box+mask 三路预测，
    直接为 mask 质量提供稠密监督信号。

    Args:
        num_stuff_classes (int):  stuff 类别数（实例分割为 0）。
        num_things_classes (int): things 类别数。
        encoder (dict):           pixel_decoder 配置。
        decoder (dict):           MaskDINODecoder 配置。
        num_query (int):          主路径解码器查询数。默认 100。
        dn_cfg (dict, optional):  DN 查询生成器配置（兼容 config 格式）。
        max_pos_coords (int):     每张图像辅助查询的最大正样本数。默认 300。
        with_aux_mask (bool):     是否计算辅助路径的 mask + dice 损失。默认 True。
                                  若 GT 不含 masks 字段（纯检测任务），可关闭。
        aux_mask_weight (float):  辅助 mask loss 权重。默认 5.0。
        aux_dice_weight (float):  辅助 dice loss 权重。默认 5.0。
        train_cfg (dict):         训练配置（含 SetCriterion 参数 + 辅助头参数，
                                   本类自动过滤后者）。
        test_cfg (dict):          测试配置。
    """

    def __init__(self,
                 num_stuff_classes: int,
                 num_things_classes: int,
                 encoder: dict,
                 decoder: dict,
                 num_query: int = 100,
                 dn_cfg: Optional[dict] = None,
                 max_pos_coords: int = 300,
                 with_aux_mask: bool = True,
                 aux_mask_weight: float = 5.0,
                 aux_dice_weight: float = 5.0,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None):

        # 过滤 train_cfg，仅保留 SetCriterion 可接受的键
        maskdino_train_cfg = None
        if train_cfg is not None:
            maskdino_train_cfg = {
                k: v for k, v in train_cfg.items() if k not in _AUX_CFG_KEYS
            }

        super().__init__(
            num_stuff_classes=num_stuff_classes,
            num_things_classes=num_things_classes,
            encoder=encoder,
            decoder=decoder,
            train_cfg=maskdino_train_cfg,
            test_cfg=test_cfg)

        self.max_pos_coords = max_pos_coords
        self.with_aux_mask = with_aux_mask
        self.aux_mask_weight = aux_mask_weight
        self.aux_dice_weight = aux_dice_weight
        self.dn_cfg = dn_cfg  # 保留备用

        # 独立的 bbox regression 损失模块（cls 复用 class_embed，mask 复用 mask_embed）
        self.aux_loss_l1 = L1Loss(loss_weight=5.0)
        self.aux_loss_giou = GIoULoss(loss_weight=2.0)

    # ================================================================== #
    # 协同辅助损失：主入口
    # ================================================================== #

    def loss_aux(self,
                 feats: Tuple[Tensor],
                 pos_coords: tuple,
                 head_idx: int,
                 batch_data_samples: list) -> dict:
        """计算协同辅助损失（cls + box + mask + dice）。

        完整流程：
          1. 运行 pixel_decoder → 得到 mask_features（1/4 分辨率） + multi_scale_feats
          2. get_aux_targets → 将 pos_coords 转化为解码器查询 + 匹配 GT masks
          3. forward_aux_decoder → cls + box + mask 完整三路预测（复用 predictor 权重）
          4. loss_aux_by_feat → 全损失计算

        Args:
            feats (tuple[Tensor]): FPN 特征图，来自 CoMaskDINO.extract_feat()。
            pos_coords (tuple):    辅助头正样本坐标包（rcnn 或 atss 格式，见 get_aux_targets）。
            head_idx (int):        第几个辅助头（用于区分损失键名）。
            batch_data_samples:    含 gt_instances（bboxes/labels/masks）的批量数据样本。

        Returns:
            dict: 辅助损失字典，含 loss_cls_aux / loss_bbox_aux / loss_iou_aux
                  及（可选）loss_mask_aux / loss_dice_aux，以及各解码层的 d{i}.* 版本。
        """
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        batch_gt_instances = [ds.gt_instances for ds in batch_data_samples]

        # Step 1: pixel_decoder 前向
        #   mask_features   : [bs, C, H/4, W/4]  供 mask_embed 使用的高分辨率特征
        #   multi_scale_feats: list[Tensor]  解码器 memory 来源
        mask_features, _enc_feats, multi_scale_feats = \
            self.pixel_decoder.forward_features(feats, None)

        # Step 2: 构建辅助查询，同时匹配 GT masks
        #   mask_features 用于确定 GT mask 的目标分辨率（与 mask_preds 保持一致）
        aux_targets = self.get_aux_targets(
            pos_coords, batch_img_metas, multi_scale_feats, head_idx,
            batch_gt_instances=batch_gt_instances if self.with_aux_mask else None,
            mask_features=mask_features)

        # Step 3: 完整解码器前向（cls + box + mask）
        all_cls_scores, all_bbox_preds, all_mask_preds = self.forward_aux_decoder(
            multi_scale_feats, mask_features, aux_targets)

        # Step 4: 计算全损失
        (aux_coords, aux_labels, aux_bbox_targets,
         label_weights, bbox_weights, _aux_feats, _attn_mask,
         aux_gt_masks, gt_mask_weights) = aux_targets

        losses = self.loss_aux_by_feat(
            all_cls_scores=all_cls_scores,
            all_bbox_preds=all_bbox_preds,
            all_mask_preds=all_mask_preds,
            aux_labels=aux_labels,
            aux_bbox_targets=aux_bbox_targets,
            label_weights=label_weights,
            bbox_weights=bbox_weights,
            aux_gt_masks=aux_gt_masks,
            gt_mask_weights=gt_mask_weights)

        return losses

    # ================================================================== #
    # 辅助查询构建 + GT Mask 匹配
    # ================================================================== #

    def get_aux_targets(self,
                        pos_coords: tuple,
                        img_metas: List[dict],
                        mlvl_feats: List[Tensor],
                        head_idx: int,
                        batch_gt_instances=None,
                        mask_features: Optional[Tensor] = None) -> tuple:
        """将辅助头正样本转化为解码器查询格式，并匹配 GT masks。

        pos_coords 格式说明：
          rcnn: (coords[bs,n,4], labels[bs,n], targets[bs,n,4], feats[bs,n,C], 'rcnn')
                coords/targets 均为绝对坐标 xyxy
          atss: (anchors[list_bs], labels[list_bs], targets[list_bs], 'atss')
                各字段均为 per-image 列表，坐标为绝对 xyxy

        GT Mask 匹配逻辑：
          对每张图的每个正样本查询，其 aux_bbox_targets 与图中某 GT 实例的 bbox 完全对应
          （正是由辅助头 assigner 分配的 GT bbox）。通过计算 aux_bbox_targets 与
          gt_instances.bboxes 的 IoU，找到最匹配的 GT 实例，取其 mask。

        Args:
            mask_features: [bs, C, H_m, W_m] 用于确定 GT mask 的目标分辨率，
                          必须与 mask_preds 的分辨率一致。若为 None，则使用 mlvl_feats[-1]。

        Returns:
            9-tuple:
              aux_coords      [bs, M, 4]    归一化 cxcywh 查询坐标（含噪声）
              aux_labels      [bs, M]       类别（bg = num_classes）
              aux_targets     [bs, M, 4]    归一化 cxcywh GT 框
              label_weights   [bs, M]       分类权重（padding=0）
              bbox_weights    [bs, M, 4]    bbox 权重（padding=0）
              aux_feats       [bs, M, C]    查询内容特征
              attn_masks      None
              aux_gt_masks    [bs, M, H_m, W_m] or None  对应 GT mask
              gt_mask_weights [bs, M]       mask 损失权重（正样本=1, padding=0）
        """
        head_name = pos_coords[-1]   # 'rcnn' or 'atss'
        coords, labels, targets = pos_coords[0], pos_coords[1], pos_coords[2]
        bs = len(img_metas)
        c = mlvl_feats[0].shape[1]

        # 统一为 per-image list 格式
        if 'rcnn' in head_name:
            roi_feats = pos_coords[3]          # [bs, n, C]
            coords_list  = [coords[i] for i in range(bs)]
            labels_list  = [labels[i] for i in range(bs)]
            targets_list = [targets[i] for i in range(bs)]
            feats_list   = [roi_feats[i] for i in range(bs)]
        else:
            # atss: already per-image lists
            coords_list  = coords
            labels_list  = labels
            targets_list = targets
            feats_list   = self._build_fpn_feats_list(mlvl_feats, c)

        # 确定 batch 内最大正样本数（min 9，max max_pos_coords）
        bg_class_ind = self.num_classes
        device = mlvl_feats[0].device
        max_num = 0
        for i in range(bs):
            label = labels_list[i]
            pos_cnt = ((label >= 0) & (label < bg_class_ind)).sum().item()
            max_num = max(max_num, pos_cnt)
        max_num = max(min(self.max_pos_coords, max_num), 9)

        label_weights = torch.ones(bs, max_num, device=device)
        bbox_weights  = torch.ones(bs, max_num, 4, device=device)
        gt_mask_weights = torch.zeros(bs, max_num, device=device)

        out_coords, out_labels, out_targets, out_feats = [], [], [], []
        out_gt_masks = [] if batch_gt_instances is not None else None

        for i in range(bs):
            coord  = coords_list[i]    # [N_i, 4] abs xyxy
            label  = labels_list[i]    # [N_i]
            target = targets_list[i]   # [N_i, 4] abs xyxy
            feat   = feats_list[i]     # [N_i, C] or [total_pos, C]

            img_h, img_w = img_metas[i]['img_shape'][:2]
            factor = coord.new_tensor([img_w, img_h, img_w, img_h])

            # 正样本筛选
            pos_inds = ((label >= 0) & (label < bg_class_ind)).nonzero(
                as_tuple=False).squeeze(1)

            # 超限随机下采样
            if len(pos_inds) > max_num:
                perm = torch.randperm(len(pos_inds), device=pos_inds.device)
                pos_inds = pos_inds[perm[:max_num]]

            num_pos = len(pos_inds)

            if num_pos > 0:
                pos_coord_norm = bbox_xyxy_to_cxcywh(
                    coord[pos_inds] / factor).clamp(0., 1.)
                pos_target_norm = bbox_xyxy_to_cxcywh(
                    target[pos_inds] / factor).clamp(0., 1.)
                pos_label = label[pos_inds]
                if 'rcnn' in head_name:
                    pos_feat = feat[pos_inds]
                else:
                    pos_feat = self._sample_fpn_feat(feat, pos_inds, len(coord))
            else:
                pos_coord_norm = coord.new_zeros(0, 4)
                pos_target_norm = coord.new_zeros(0, 4)
                pos_label = label.new_zeros(0)
                pos_feat = coord.new_zeros(0, c)

            # ---- Padding ----
            pad = max_num - num_pos
            label_weights[i, num_pos:] = 0.
            bbox_weights[i, num_pos:]  = 0.
            gt_mask_weights[i, :num_pos] = 1.

            bg_label = label.new_full((pad,), bg_class_ind)
            pad_coord  = coord.new_zeros(pad, 4)
            pad_target = coord.new_zeros(pad, 4)
            pad_feat   = coord.new_zeros(pad, c)

            out_coords.append(torch.cat([pos_coord_norm, pad_coord]))
            out_labels.append(torch.cat([pos_label, bg_label]))
            out_targets.append(torch.cat([pos_target_norm, pad_target]))
            out_feats.append(torch.cat([pos_feat, pad_feat]))

            # ---- GT Mask 匹配 ----
            if batch_gt_instances is not None:
                gt_inst = batch_gt_instances[i]
                # 使用 mask_features 的分辨率（与 mask_preds 一致）
                if mask_features is not None:
                    mask_h, mask_w = mask_features.shape[-2], mask_features.shape[-1]
                else:
                    mask_h, mask_w = mlvl_feats[-1].shape[-2], mlvl_feats[-1].shape[-1]
                pos_gt_masks = self._match_gt_masks(
                    pos_target_norm, gt_inst, img_h, img_w,
                    mask_h=mask_h, mask_w=mask_w)
                out_gt_masks.append(pos_gt_masks)   # [num_pos, H_m, W_m] or None

        aux_coords  = torch.stack(out_coords)    # [bs, M, 4]
        aux_labels  = torch.stack(out_labels)    # [bs, M]
        aux_targets = torch.stack(out_targets)   # [bs, M, 4]
        aux_feats   = torch.stack(out_feats)     # [bs, M, C]

        # ---- 将 per-image GT masks 整合为 batch tensor ----
        aux_gt_masks_t = None
        if batch_gt_instances is not None and any(
                m is not None for m in out_gt_masks):
            # 获取 mask 分辨率（必须与 mask_preds 一致，来自 mask_features）
            # mask_features: [bs, C, H_m, W_m]，用于生成 mask_preds
            if mask_features is not None:
                mh, mw = mask_features.shape[-2], mask_features.shape[-1]
            else:
                # 回退：使用最后一级 FPN 特征（可能与 mask_preds 不匹配）
                mh = mlvl_feats[-1].shape[-2]
                mw = mlvl_feats[-1].shape[-1]
            # 构建 [bs, M, mh, mw] 零 tensor，再填入有效 mask
            aux_gt_masks_t = aux_coords.new_zeros(bs, max_num, mh, mw)
            for i, pos_gt_masks in enumerate(out_gt_masks):
                if pos_gt_masks is not None and pos_gt_masks.shape[0] > 0:
                    num_pos = pos_gt_masks.shape[0]
                    aux_gt_masks_t[i, :num_pos] = pos_gt_masks

        return (aux_coords, aux_labels, aux_targets,
                label_weights, bbox_weights, aux_feats, None,
                aux_gt_masks_t, gt_mask_weights)

    def _match_gt_masks(self,
                        pos_target_norm: Tensor,
                        gt_inst,
                        img_h: int, img_w: int,
                        mask_h: int, mask_w: int) -> Optional[Tensor]:
        """为每个正样本查询匹配对应的 GT mask，并下采样到 (mask_h, mask_w)。

        匹配逻辑：pos_target_norm 是辅助头 assigner 分配的 GT bbox（cxcywh 归一化）。
        将其转回绝对 xyxy 坐标后与 gt_inst.bboxes（绝对 xyxy）计算 IoU，
        取 IoU 最高的 GT 实例作为匹配结果。

        Args:
            pos_target_norm: [num_pos, 4] 归一化 cxcywh GT bbox
            gt_inst:         InstanceData，含 bboxes([N,4] xyxy), masks
            img_h, img_w:    原图尺寸
            mask_h, mask_w:  目标 mask 输出分辨率（等于 pixel_decoder 最大尺度特征图）

        Returns:
            Tensor [num_pos, mask_h, mask_w] 或 None（若无 mask 字段）
        """
        if not hasattr(gt_inst, 'masks') or len(gt_inst.masks) == 0:
            return None
        if pos_target_norm.shape[0] == 0:
            return pos_target_norm.new_zeros(0, mask_h, mask_w)

        device = pos_target_norm.device
        factor = pos_target_norm.new_tensor([img_w, img_h, img_w, img_h])

        # 将 aux 目标框转回绝对 xyxy
        pos_tgt_abs = bbox_cxcywh_to_xyxy(pos_target_norm * factor)  # [P, 4]
        gt_bboxes = gt_inst.bboxes.to(device)  # [N, 4] xyxy abs

        # IoU 匹配：每个 aux 查询对应 IoU 最大的 GT
        iou_mat = bbox_overlaps(pos_tgt_abs, gt_bboxes)  # [P, N]
        gt_idx = iou_mat.argmax(dim=1)                    # [P]

        # 取 GT masks（BitmapMasks 或 PolygonMasks → 转 bool tensor）
        gt_masks_all = gt_inst.masks  # BitmapMasks: masks.masks [N, H, W]
        if hasattr(gt_masks_all, 'masks'):
            masks_tensor = torch.from_numpy(
                gt_masks_all.masks).float().to(device)  # [N, H_gt, W_gt]
        else:
            # PolygonMasks → to_bitmap
            masks_tensor = torch.from_numpy(
                gt_masks_all.to_bitmap().masks).float().to(device)

        # 按 gt_idx 取出正样本对应的 GT mask，并 resize 到目标分辨率
        sel_masks = masks_tensor[gt_idx]  # [P, H_gt, W_gt]
        sel_masks = F.interpolate(
            sel_masks.unsqueeze(1),
            size=(mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(1)  # [P, mask_h, mask_w]

        return sel_masks  # [num_pos, mask_h, mask_w]

    # ================================================================== #
    # 辅助解码器前向：cls + box + mask 三路完整预测
    # ================================================================== #

    def forward_aux_decoder(self,
                             multi_scale_feats: List[Tensor],
                             mask_features: Tensor,
                             aux_targets: tuple) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """完整辅助解码器前向：复用 predictor 所有权重，输出 cls + box + mask。

        与主路径（MaskDINODecoder.forward）的关键异同：
          相同：共享 input_proj / TransformerDecoder / class_embed / bbox_embed / mask_embed
          不同：
            - query 初始化来自辅助头正样本（而非 two-stage top-k 候选）
            - 没有 DN 噪声查询
            - valid_ratios 从 dummy-full mask 计算（无 padding mask 变体）

        Tensor 格式约定（与 MaskDINODecoder 内部一致）：
          解码器内部：tgt / memory / refpoints 均使用 [M, bs, *] 序列优先格式
          forward_prediction_heads 输入：[bs, M, C] 批次优先格式（transpose(0,1)）

        Args:
            multi_scale_feats: pixel_decoder 输出的多尺度特征，list of [bs, C, H_i, W_i]
            mask_features:     pixel_decoder 输出的 1/4 分辨率 mask 特征 [bs, C, H/4, W/4]
            aux_targets:       get_aux_targets() 的 9-tuple 返回值

        Returns:
            all_cls_scores  [L, bs, M, num_classes]
            all_bbox_preds  [L, bs, M, 4]            归一化 cxcywh
            all_mask_preds  [L, bs, M, H/4, W/4] or None  （仅最后层预测 or 全部层）
        """
        from mmdet.models.layers.transformer import inverse_sigmoid

        (aux_coords, aux_labels, aux_bbox_targets,
         label_weights, bbox_weights, aux_feats,
         _attn_masks, aux_gt_masks, gt_mask_weights) = aux_targets

        predictor = self.predictor   # MaskDINODecoder instance
        bs, num_q, c = aux_feats.shape
        device = aux_feats.device

        # ------------------------------------------------------------------ #
        # Step 1: 构建 memory（multi-scale feats → flatten）
        # 与 MaskDINODecoder.forward() 的 memory 构建保持完全一致：
        #   - 使用 predictor.input_proj 做 channel projection
        #   - 按 predictor.num_feature_levels 决定总级数
        #   - 超出 pixel_decoder 输出的级数做 avg_pool 补充
        # ------------------------------------------------------------------ #
        num_levels = predictor.num_feature_levels
        src_list, spatial_shapes, mask_list = [], [], []

        for lvl in range(min(len(multi_scale_feats), num_levels)):
            src = multi_scale_feats[lvl]                  # [bs, C, H, W]
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            src = predictor.input_proj[lvl](src)          # identity or conv [bs, C, H, W]
            mask_list.append(src.new_zeros(bs, h, w, dtype=torch.bool))  # all valid
            src_list.append(src.flatten(2).transpose(1, 2))              # [bs, H*W, C]

        for lvl in range(len(multi_scale_feats), num_levels):
            prev = multi_scale_feats[-1]
            src = F.avg_pool2d(prev, kernel_size=2, stride=2, padding=0)
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            src = predictor.input_proj[lvl](src)
            mask_list.append(src.new_zeros(bs, h, w, dtype=torch.bool))
            src_list.append(src.flatten(2).transpose(1, 2))

        memory = torch.cat(src_list, dim=1).permute(1, 0, 2)  # [N_total, bs, C]
        spatial_shapes_t = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat([
            spatial_shapes_t.new_zeros((1,)),
            spatial_shapes_t.prod(1).cumsum(0)[:-1]])
        valid_ratios = torch.stack(
            [predictor.get_valid_ratio(m) for m in mask_list], dim=1)  # [bs, nlevels, 2]

        # ------------------------------------------------------------------ #
        # Step 2: 初始化 query
        #   tgt (content):          辅助头正样本特征，序列优先 [M, bs, C]
        #   refpoints_unsigmoid:    inverse_sigmoid(正样本坐标)，[M, bs, 4]
        # ------------------------------------------------------------------ #
        pos_anchors = aux_coords.clamp(1e-4, 1 - 1e-4)               # [bs, M, 4]
        refpoints_unsigmoid = inverse_sigmoid(pos_anchors)            # [bs, M, 4]

        tgt = aux_feats.permute(1, 0, 2)                              # [M, bs, C]
        refpoints_unsigmoid_seq = refpoints_unsigmoid.permute(1, 0, 2)  # [M, bs, 4]

        # ------------------------------------------------------------------ #
        # Step 3: TransformerDecoder 前向
        #   hs         [L, M, bs, C]      每层输出，已经过 decoder.norm（单次）
        #   ref_points [L+1, M, bs, 4]    sigmoid 空间，初始 + 每层 refine
        # ------------------------------------------------------------------ #
        hs, ref_points = predictor.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=None,
            pos=None,
            refpoints_unsigmoid=refpoints_unsigmoid_seq,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes_t,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            bbox_embed=predictor.bbox_embed)

        # ------------------------------------------------------------------ #
        # Step 4: 对每解码层分别计算 cls + box + mask
        #
        # 分类（class_embed）:
        #   复用 predictor.forward_prediction_heads() 的行为——
        #   接受 [bs, M, C]（已单次 norm），再施加一次 decoder.norm（双重 norm），
        #   然后由 class_embed 产生 logits。
        #   这与主路径 forward_prediction_heads 完全一致，权重共享有意义。
        #
        # Mask（mask_embed）:
        #   直接调用 predictor.forward_prediction_heads(output_batch_first, mask_features)
        #   mask_features 来自 pixel_decoder，分辨率 1/4，与主路径完全一致。
        #
        # 回归（bbox_embed）:
        #   iterative refinement：delta = bbox_embed[i](layer_hs) + inverse_sigmoid(ref_points[i])
        # ------------------------------------------------------------------ #
        all_cls_scores, all_bbox_preds, all_mask_preds = [], [], []

        for i in range(len(hs)):
            layer_hs = hs[i]           # [M, bs, C]，已单次 norm
            layer_hs_batch = layer_hs.permute(1, 0, 2)  # [bs, M, C]

            # ---- 分类 + Mask：forward_prediction_heads ----
            outputs_class, outputs_mask = predictor.forward_prediction_heads(
                layer_hs_batch, mask_features,
                pred_mask=(i == len(hs) - 1 or self.with_aux_mask))
            # outputs_class: [bs, M, num_classes]
            # outputs_mask:  [bs, M, H/4, W/4]  或 None

            # ---- Box regression：iterative refinement ----
            ref_sig = ref_points[i]                               # [M, bs, 4]
            delta   = predictor.bbox_embed[i](layer_hs)           # [M, bs, 4]
            box     = (delta + inverse_sigmoid(ref_sig)).sigmoid() # [M, bs, 4]
            box     = box.permute(1, 0, 2)                        # [bs, M, 4]

            all_cls_scores.append(outputs_class)
            all_bbox_preds.append(box)
            all_mask_preds.append(outputs_mask)

        all_cls_scores = torch.stack(all_cls_scores)  # [L, bs, M, num_classes]
        all_bbox_preds = torch.stack(all_bbox_preds)  # [L, bs, M, 4]

        # mask preds: None 元素（非末层）不堆叠，保持列表格式
        # 后续 loss 只用最后一层 mask（或所有层，取决于 with_aux_mask 配置）
        mask_preds_out = all_mask_preds  # list of [bs, M, H, W] or None

        return all_cls_scores, all_bbox_preds, mask_preds_out

    # ================================================================== #
    # 辅助损失计算
    # ================================================================== #

    def loss_aux_by_feat(self,
                         all_cls_scores: Tensor,
                         all_bbox_preds: Tensor,
                         all_mask_preds: list,
                         aux_labels: Tensor,
                         aux_bbox_targets: Tensor,
                         label_weights: Tensor,
                         bbox_weights: Tensor,
                         aux_gt_masks: Optional[Tensor],
                         gt_mask_weights: Tensor) -> dict:
        """对每个解码层分别计算辅助 cls + box + mask + dice 损失。

        mask/dice 损失仅在最后一层计算（forward_prediction_heads 在非末层
        可能不输出 mask_preds，通过 pred_mask 参数控制）。
        若需要所有层都计算 mask loss，可在 forward_aux_decoder 中对所有层
        设置 pred_mask=True，并在此处遍历全部层。

        Args:
            all_cls_scores  [L, bs, M, num_classes]
            all_bbox_preds  [L, bs, M, 4]            cxcywh [0,1]
            all_mask_preds  list[L] of [bs, M, H, W] or None
            aux_labels      [bs, M]
            aux_bbox_targets [bs, M, 4]
            label_weights   [bs, M]
            bbox_weights    [bs, M, 4]
            aux_gt_masks    [bs, M, H_m, W_m] or None  GT mask（已 resize）
            gt_mask_weights [bs, M]            mask loss 权重（positive=1, padding=0）

        Returns:
            dict: 损失字典
        """
        num_dec_layers = all_cls_scores.shape[0]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self._loss_aux_single,
            all_cls_scores,
            all_bbox_preds,
            [aux_labels] * num_dec_layers,
            [aux_bbox_targets] * num_dec_layers,
            [label_weights] * num_dec_layers,
            [bbox_weights] * num_dec_layers)

        loss_dict = {}
        # 最后一层（主要监督）
        loss_dict['loss_cls_aux']  = losses_cls[-1]
        loss_dict['loss_bbox_aux'] = losses_bbox[-1]
        loss_dict['loss_iou_aux']  = losses_iou[-1]
        # 前面各层（深度监督）
        for dec_lid, (lc, lb, li) in enumerate(
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])):
            loss_dict[f'd{dec_lid}.loss_cls_aux']  = lc
            loss_dict[f'd{dec_lid}.loss_bbox_aux'] = lb
            loss_dict[f'd{dec_lid}.loss_iou_aux']  = li

        # ---- Mask + Dice 损失（最后一层，或所有有 mask 预测的层）----
        if self.with_aux_mask and aux_gt_masks is not None:
            for layer_idx, mask_pred in enumerate(all_mask_preds):
                if mask_pred is None:
                    continue
                loss_mask, loss_dice = self._loss_aux_mask(
                    mask_pred, aux_gt_masks, gt_mask_weights)
                prefix = '' if layer_idx == num_dec_layers - 1 else f'd{layer_idx}.'
                loss_dict[f'{prefix}loss_mask_aux'] = (
                    loss_mask * self.aux_mask_weight)
                loss_dict[f'{prefix}loss_dice_aux'] = (
                    loss_dice * self.aux_dice_weight)

        return loss_dict

    def _loss_aux_single(self,
                         cls_scores: Tensor,
                         bbox_preds: Tensor,
                         aux_labels: Tensor,
                         aux_bbox_targets: Tensor,
                         label_weights: Tensor,
                         bbox_weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """单个解码层的辅助 cls + bbox + iou 损失。

        分类：BCE with IoU-quality soft labels，屏蔽 padding 位置。
        回归：L1 + GIoU，仅对前景正样本计算。

        Returns:
            (loss_cls, loss_bbox, loss_iou)
        """
        bs, num_q, num_cls = cls_scores.shape
        bg_class_ind = self.num_classes

        cls_scores_flat  = cls_scores.reshape(-1, num_cls)
        bbox_preds_flat  = bbox_preds.reshape(-1, 4)
        labels_flat      = aux_labels.reshape(-1)
        targets_flat     = aux_bbox_targets.reshape(-1, 4)
        lw_flat          = label_weights.reshape(-1)
        bw_flat          = bbox_weights.reshape(-1, 4)

        fg_mask  = (labels_flat >= 0) & (labels_flat < bg_class_ind)
        num_pos  = max(fg_mask.sum().item(), 1)

        # ---- Classification: BCE with IoU-quality soft labels ----
        target_soft = cls_scores_flat.new_zeros(cls_scores_flat.shape)
        if fg_mask.any():
            iou_scores = bbox_overlaps(
                bbox_cxcywh_to_xyxy(bbox_preds_flat[fg_mask].detach()),
                bbox_cxcywh_to_xyxy(targets_flat[fg_mask]),
                is_aligned=True).clamp(0.)
            target_soft[fg_mask, labels_flat[fg_mask]] = iou_scores

        loss_cls = (
            F.binary_cross_entropy_with_logits(
                cls_scores_flat, target_soft, reduction='none'
            ).sum(-1) * lw_flat
        ).sum() / (num_pos * num_cls + 1)

        # ---- Regression: L1 + GIoU（前景正样本）----
        if fg_mask.any():
            pos_pred = bbox_preds_flat[fg_mask]
            pos_tgt  = targets_flat[fg_mask]
            pos_bw   = bw_flat[fg_mask]

            loss_bbox = self.aux_loss_l1(
                pos_pred, pos_tgt, weight=pos_bw, avg_factor=num_pos)
            loss_iou  = self.aux_loss_giou(
                bbox_cxcywh_to_xyxy(pos_pred),
                bbox_cxcywh_to_xyxy(pos_tgt),
                weight=pos_bw[:, 0], avg_factor=num_pos)
        else:
            loss_bbox = bbox_preds_flat.sum() * 0.
            loss_iou  = bbox_preds_flat.sum() * 0.

        return loss_cls, loss_bbox, loss_iou

    def _loss_aux_mask(self,
                       mask_preds: Tensor,
                       gt_masks: Tensor,
                       mask_weights: Tensor) -> Tuple[Tensor, Tensor]:
        """辅助路径的 mask 损失（sigmoid-focal BCE + Dice）。

        使用点采样（与 MaskDINO SetCriterion 一致）降低计算量：
        在 GT mask 上按重要性采样若干点，仅在这些点计算损失。

        Args:
            mask_preds   [bs, M, H, W]  预测 mask logits
            gt_masks     [bs, M, H, W]  GT binary masks（已 resize）
            mask_weights [bs, M]        权重（正样本=1, padding=0）

        Returns:
            (loss_mask_raw, loss_dice_raw)  均为标量 Tensor，
            调用方乘以 aux_mask_weight / aux_dice_weight
        """
        bs, M, H, W = mask_preds.shape
        num_pos = max(mask_weights.sum().item(), 1)

        # 断言：确保 mask_preds 和 gt_masks 空间维度一致
        # 如果不一致，说明 get_aux_targets 中使用了错误的分辨率
        assert gt_masks.shape[-2:] == (H, W), \
            f"Spatial dimension mismatch: mask_preds={mask_preds.shape}, gt_masks={gt_masks.shape}"

        # 重塑为 [bs*M, 1, H, W] 进行点采样
        mask_preds_flat = mask_preds.reshape(bs * M, 1, H, W)
        gt_masks_flat   = gt_masks.reshape(bs * M, 1, H, W)

        # 重要性点采样（与 MaskDINO SetCriterion 一致）
        num_points = self.criterion.num_points \
            if hasattr(self, 'criterion') else 12544
        num_points = min(num_points, H * W)

        with torch.no_grad():
            # 基于预测的不确定性采样（高不确定度点更关键）
            pred_prob = mask_preds_flat.sigmoid()
            uncertainty = -(pred_prob - 0.5).abs()  # 越接近 0.5 越不确定
            # 加权随机采样：重要性高的点被优先选中
            pt_coords = torch.rand(
                bs * M, 1, num_points, 2, device=mask_preds.device) * 2 - 1
            # 从 uncertainty map 中取采样点的值，作为重要性权重
            importance = F.grid_sample(
                uncertainty, pt_coords, align_corners=False).squeeze(1).squeeze(1)
            # 75% 重要性采样 + 25% 均匀采样（与 SetCriterion 的 3:1 策略一致）
            n_important = int(0.75 * num_points)
            n_uniform   = num_points - n_important
            topk_idx = importance.topk(n_important, dim=1)[1]  # [bs*M, k_imp]
            rand_idx  = torch.randint(
                H * W, (bs * M, n_uniform), device=importance.device)
            # 转为归一化坐标
            h_coords_topk = (topk_idx // W).float() / H * 2 - 1
            w_coords_topk = (topk_idx  % W).float() / W * 2 - 1
            h_coords_rand = (rand_idx  // W).float() / H * 2 - 1
            w_coords_rand = (rand_idx  % W).float() / W * 2 - 1
            sample_coords = torch.stack([
                torch.cat([w_coords_topk, w_coords_rand], dim=1),
                torch.cat([h_coords_topk, h_coords_rand], dim=1)
            ], dim=-1).unsqueeze(1)  # [bs*M, 1, num_points, 2]

        # 在采样点处取预测值和 GT 值
        pred_pts = F.grid_sample(
            mask_preds_flat, sample_coords,
            mode='bilinear', align_corners=False
        ).squeeze(1).squeeze(1)  # [bs*M, num_points]
        gt_pts = F.grid_sample(
            gt_masks_flat, sample_coords,
            mode='nearest', align_corners=False
        ).squeeze(1).squeeze(1)  # [bs*M, num_points]

        # 展平权重 [bs*M]
        wt_flat = mask_weights.reshape(bs * M)

        # Binary Focal-like BCE：每个点都计算，再 reduce
        bce = F.binary_cross_entropy_with_logits(
            pred_pts, gt_pts, reduction='none')  # [bs*M, num_points]
        loss_mask = (bce.mean(dim=1) * wt_flat).sum() / num_pos

        # Dice loss（在 sigmoid 概率上计算）
        pred_sig  = pred_pts.sigmoid()                            # [bs*M, num_pts]
        numerator = 2 * (pred_sig * gt_pts).sum(dim=1)            # [bs*M]
        denominator = pred_sig.sum(dim=1) + gt_pts.sum(dim=1)     # [bs*M]
        dice_score = 1 - (numerator + 1) / (denominator + 1)      # [bs*M]
        loss_dice = (dice_score * wt_flat).sum() / num_pos

        return loss_mask, loss_dice

    # ================================================================== #
    # 辅助工具：FPN 特征提取（用于 ATSS 无 ROI 特征的情况）
    # ================================================================== #

    def _build_fpn_feats_list(self,
                               mlvl_feats: List[Tensor],
                               c: int) -> List[Tensor]:
        """将 FPN 多尺度特征图展平，构建 per-image 空间特征列表（ATSS 使用）。

        Returns:
            list of Tensor: 每个元素 [total_spatial_positions, C]
        """
        bs = mlvl_feats[0].shape[0]
        feats_list = []
        for i in range(bs):
            per_img = [
                feat[i].reshape(c, -1).transpose(0, 1)
                for feat in mlvl_feats
            ]
            feats_list.append(torch.cat(per_img, dim=0))  # [total, C]
        return feats_list

    def _sample_fpn_feat(self,
                          flat_feats: Tensor,
                          pos_inds: Tensor,
                          num_anchors: int) -> Tensor:
        """将 anchor 索引映射到 FPN 展平特征的空间位置（ATSS 使用）。

        Args:
            flat_feats:   [total_spatial, C]  展平后的 FPN 特征
            pos_inds:     [num_pos]  anchor 索引
            num_anchors:  该图总 anchor 数

        Returns:
            Tensor: [num_pos, C]
        """
        total_spatial = flat_feats.shape[0]
        ratio = max(1, num_anchors // total_spatial)
        spatial_inds = (pos_inds // ratio).clamp(0, total_spatial - 1)
        return flat_feats[spatial_inds]
