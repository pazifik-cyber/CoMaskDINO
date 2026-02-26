import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init
from torch.nn.init import normal_

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.registry import MODELS

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

# In order to save the cost and effort of reproduction,
# I did not refactor it into the style of mmdet 3.x DETR.


class Transformer(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(Transformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory


@MODELS.register_module(force=True)
class DeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(DeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@MODELS.register_module(force=True)
class DeformableDetrTransformer(Transformer):
    """实现DeformableDETR transformer。

    Args:
        as_two_stage (bool): 从编码器特征生成查询。
            默认值: False。
        num_feature_levels (int): 来自FPN的特征图数量:
            默认值: 4。
        two_stage_num_proposals (int): 当设置
            `as_two_stage`为True时的提案数量。默认值: 300。
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(DeformableDetrTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage  # 是否使用两阶段
        self.num_feature_levels = num_feature_levels  # 特征层级数量
        self.two_stage_num_proposals = two_stage_num_proposals  # 两阶段提案数量
        self.embed_dims = self.encoder.embed_dims  # 嵌入维度
        self.init_layers()

    def init_layers(self):
        """初始化DeformableDetrTransformer的层。"""
        # 每个特征层级的嵌入参数
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            # 编码器输出的线性层
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            # 位置变换层
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            # 单阶段时的参考点生成层
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """初始化transformer权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """从编码的记忆中生成提案。

        Args:
            memory (Tensor) : 编码器的输出，
                形状为 (bs, num_key, embed_dim)。num_key等于
                所有层级特征图上的点数量。
            memory_padding_mask (Tensor): 记忆的填充掩码。
                形状为 (bs, num_key)。
            spatial_shapes (Tensor): 所有特征图的形状。
                形状为 (num_level, 2)。

        Returns:
            tuple: 特征图和边界框预测的元组。

                - output_memory (Tensor): 解码器的输入，  \
                    形状为 (bs, num_key, embed_dim)。num_key等于 \
                    所有层级特征图上的点数量。
                - output_proposals (Tensor): 经过反sigmoid的归一化提案， \
                    形状为 (bs, num_keys, 4)。
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        # 为每个特征层级生成提案
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            # 生成网格坐标
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # 归一化坐标
            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            # 设置边界框的宽高
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        
        output_proposals = torch.cat(proposals, 1)
        # 筛选有效提案
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        # 对数变换
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        # 处理输出记忆
        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """获取解码器中使用的参考点。

        Args:
            spatial_shapes (Tensor): 所有特征图的形状，
                形状为 (num_level, 2)。
            valid_ratios (Tensor): 特征图上有效点的比例，
                形状为 (bs, num_levels, 2)
            device (obj:`device`): 参考点应该在的设备。

        Returns:
            Tensor: 解码器中使用的参考点，形状为 \
                (bs, num_keys, num_levels, 2)。
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 为每个层级生成参考点网格
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            # 归一化参考点坐标
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # 应用有效比例
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """获取所有层级特征图的有效比例。"""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """获取提案的位置嵌入。"""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Transformer的前向函数。

        Args:
            mlvl_feats (list(Tensor)): 来自不同层级的输入查询。
                每个元素形状为 [bs, embed_dims, h, w]。
            mlvl_masks (list(Tensor)): 来自不同层级用于编码器和解码器的
                key_padding_mask，每个元素形状为 [bs, h, w]。
            query_embed (Tensor): 解码器的查询嵌入，
                形状为 [num_query, c]。
            mlvl_pos_embeds (list(Tensor)): 来自不同层级特征的位置编码，
                形状为 [bs, embed_dims, h, w]。
            reg_branches (obj:`nn.ModuleList`): 每个解码器层特征图的回归头。
                只有在 `with_box_refine` 为True时才会传递。默认为None。
            cls_branches (obj:`nn.ModuleList`): 每个解码器层特征图的分类头。
                只有在 `as_two_stage` 为True时才会传递。默认为None。


        Returns:
            tuple[Tensor]: 包含以下张量的解码器结果。

                - inter_states: 解码器的输出。如果
                    return_intermediate_dec为True，输出形状为 \
                      (num_dec_layers, bs, num_query, embed_dims)，否则形状为 \
                      (1, bs, num_query, embed_dims)。
                - init_reference_out: 参考点的初始值， \
                    形状为 (bs, num_queries, 4)。
                - inter_references_out: 解码器中参考点的内部值， \
                    形状为 (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: 从编码器特征图生成的提案的分类分数， \
                    形状为 (batch, h*w, num_classes)。 \
                    只有在 `as_two_stage` 为True时才返回，否则为None。
                - enc_outputs_coord_unact: 从编码器特征图生成的回归结果， \
                    形状为 (batch, h*w, 4)。只有在 `as_two_stage` 为True时 \
                    才返回，否则为None。
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 处理多层级特征
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # 展平特征
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # 添加层级嵌入
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        
        # 拼接所有层级的特征
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # 获取参考点
        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        # 调整维度顺序用于编码器
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        
        # 编码器前向传播
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        
        if self.as_two_stage:
            # 两阶段：从编码器输出生成提案
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            # 编码器输出的分类和回归
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            # 选择前景分数最高的提案
            # 我们只使用enc_outputs_class中的第一个通道作为前景，
            # 其他(num_classes - 1)个通道实际上没有使用。
            # 它们的目标被设置为0，表示第一个类（前景），
            # 因为我们使用[0, num_classes - 1]来表示类标签，
            # 背景类由num_classes表示（类似RPN中的约定）。
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            # 生成查询的位置嵌入
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            # 单阶段：使用预定义的查询嵌入
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # 解码器前向传播
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        return inter_states, init_reference_out, \
            inter_references_out, None, None


@MODELS.register_module()
class CoDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 return_intermediate=False,
                 look_forward_twice=False,
                 **kwargs):

        super(CoDeformableDetrTransformerDecoder,
              self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points if self.
                    look_forward_twice else reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@MODELS.register_module()
class CoDeformableDetrTransformer(DeformableDetrTransformer):

    def __init__(self,
                 mixed_selection=True,
                 with_pos_coord=True,
                 with_coord_feat=True,
                 num_co_heads=1,
                 **kwargs):
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(CoDeformableDetrTransformer, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the CoDeformableDetrTransformer."""
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                # bug: this code should be 'self.head_pos_embed =
                # nn.Embedding(self.num_co_heads, self.embed_dims)',
                # we keep this bug for reproducing our results with ResNet-50.
                # You can fix this bug when reproducing results with
                # swin transformer.
                self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1,
                                                   self.embed_dims)
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(
                        nn.Linear(self.embed_dims * 2, self.embed_dims * 2))
                    self.aux_pos_trans_norm.append(
                        nn.LayerNorm(self.embed_dims * 2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(
                            nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(
                            nn.LayerNorm(self.embed_dims))

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                return_encoder_output=False,
                attn_masks=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk = query_embed.shape[0]
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            if return_encoder_output:
                return inter_states, init_reference_out,\
                    inter_references_out, enc_outputs_class,\
                    enc_outputs_coord_unact, memory
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        if return_encoder_output:
            return inter_states, init_reference_out, \
                inter_references_out, None, None, memory
        return inter_states, init_reference_out, \
            inter_references_out, None, None

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk_coords_unact = inverse_sigmoid(pos_anchors)
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](
                    self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](
                    self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, \
            inter_references_out


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


@MODELS.register_module()
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            pos_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000**(2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@MODELS.register_module()
class CoDinoTransformer(CoDeformableDetrTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDinoTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)

    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(
                        nn.Linear(self.embed_dims * 2, self.embed_dims))
                    self.aux_pos_trans_norm.append(
                        nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(
                            nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(
                            nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.query_embed.weight.data)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out, \
            topk_score, topk_anchor, memory

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        """为 CO-DETR 的辅助头（co-heads）提供的前向传播路径。

        这个方法与主干的 `forward` 方法有本质区别：
        1. 它 **跳过** 了计算成本高昂的 Encoder self-attention 过程。
        2. 它直接使用多尺度特征图作为 `memory`。
        3. 它的 `query` 和 `reference_points` 直接由输入的正样本 anchor
           (`pos_anchors`) 和其特征 (`pos_feats`) 生成。
        这使得每个辅助头都能以较低的成本，专注于对特定样本（如正样本）进行解码，
        从而实现协同训练。

        Args:
            mlvl_feats (list(Tensor)): 来自 FPN 的多尺度特征图。
            mlvl_masks (list(Tensor)): 对应特征图的 padding mask。
            pos_anchors (Tensor): 正样本 anchor，形状为 (bs, num_pos, 4)。
                它将直接作为解码器的 reference_points。
            pos_feats (Tensor, optional): 正样本对应的特征，由主干 Encoder
                的 memory 中提取。形状为 (bs, num_pos, C)。
            reg_branches (nn.ModuleList): 用于迭代式框回归的 MLP 头。
            head_idx (int): 当前是第几个辅助头，用于选择对应的变换层。

        Returns:
            tuple[Tensor]: 解码器输出。
                - inter_states: 中间层输出的 content feature。
                - inter_references_out: 中间层 refine 后的 reference points。
        """
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        # 1. 将多尺度特征图展平，与主干`forward`方法中的处理方式相同。
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        # 2. 核心区别：直接使用展平后的特征作为 memory，完全跳过 Encoder。
        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        # 3. 直接使用输入的正样本 anchor 作为 reference points。
        #    先用 inverse_sigmoid 将其从 [0,1] 转换到 logit 空间。
        topk_coords_unact = inverse_sigmoid(pos_anchors)
        reference_points = pos_anchors
        
        # 4. 根据 anchor 和特征生成解码器查询(query)。
        if self.num_co_heads > 0:
            # 4.1 将 anchor 坐标转换为正余弦位置编码。
            pos_embed = self.get_proposal_pos_embed(topk_coords_unact)
            # 4.2 使用当前辅助头专属的 MLP 和 LayerNorm 将位置编码转换为初始 query。
            #     这里的 query 同时包含了位置和内容的初始信息。
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](pos_embed))
            query = pos_trans_out
            
            # 4.3 (可选) 如果配置了 with_coord_feat，则将输入的正样本特征
            #     也通过一个专属 MLP 进行变换，并加到 query 上，以注入更丰富的内容信息。
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](
                    self.pos_feats_trans[head_idx](pos_feats))

        # 5. 调用解码器进行解码。
        #    注意：这里的 query 是动态生成的，而 key/value 是跳过 Encoder 的 memory。
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,  # key 和 value 将会复用 memory
            value=memory,
            attn_masks=None, # 辅助头解码通常不需要 attention mask
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        # 6. 返回解码结果，供辅助头计算损失。
        return inter_states, inter_references_out


@MODELS.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 with_cp=-1,
                 **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None
        self.with_cp = with_cp
        if self.with_cp > 0:
            if checkpoint_wrapper is None:
                warnings.warn('If you want to reduce GPU memory usage, \
                              please install fairscale by executing the \
                              following command: pip install fairscale.')
                return
            for i in range(self.with_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])


@MODELS.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
