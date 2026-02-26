_base_ = [
    '_base_/office_panoptic.py',
    '../../../configs/_base_/default_runtime.py'
]  # TODO: mmdet::

custom_imports = dict(
    imports=['projects.MaskDINO.vit_adapter', 'projects.MaskDINO.maskdino', 
             'projects.VideoKNet.models', 
             'projects.VideoKNet.datasets'], allow_failed_imports=False)


image_size = (512, 512)
dec_layers = 9  # decoder layers
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments)

num_things_classes = 48
num_stuff_classes = 13
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='MaskDINO',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BEiTAdapter',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),
    panoptic_head=dict(
        type='MaskDINOHead',
        num_stuff_classes=num_stuff_classes,
        num_things_classes=num_things_classes,
        encoder=dict(
            in_channels=[1024, 1024, 1024, 1024],
            in_strides=[4, 8, 16, 32],
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm_cfg=dict(type='GN', num_groups=32),
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high'),
        decoder=dict(
            in_channels=256,
            num_classes=num_things_classes + num_stuff_classes,
            hidden_dim=256,
            num_queries=300,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            mask_dim=256,
            enforce_input_project=False,
            two_stage=True,
            dn='seg',
            noise_scale=0.4,
            dn_num=100,
            # initialize_box_type='no',
            initialize_box_type='mask2box',  # diff
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels=4,
            dropout=0.0,
            activation='relu',
            nhead=8,
            dec_n_points=4,
            mask_classification=True,
            return_intermediate_dec=True,
            query_dim=4,
            dec_layer_share=False,
            semantic_ce_loss=False)),
    panoptic_fusion_head=dict(
        type='MaskDINOFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,  # MaskDINOFusionHead has no training loss
        init_cfg=None),  # MaskDINOFusionHead has no module
    train_cfg=dict(  # corresponds to SetCriterion
        num_classes=num_things_classes + num_stuff_classes,
        matcher=dict(
            cost_class=4.0,
            cost_box=5.0,
            cost_giou=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=12544),
        class_weight=4.0,
        box_weight=5.0,
        giou_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        dn='seg',
        dec_layers=dec_layers,
        box_loss=True,
        two_stage=True,
        eos_coef=0.1,
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        semantic_ce_loss=False,
        panoptic_on=False,  # TODO: Why?
        deep_supervision=True),
    test_cfg=dict(
        panoptic_on=True,
        instance_on=True,
        semantic_on=False,
        panoptic_postprocess_cfg=dict(
            object_mask_thr=0.25,  # 0.8 for MaskFormer
            iou_thr=0.8,
            filter_low_score=
            True,  # it will filter mask area where score is less than 0.5.
            panoptic_temperature=0.06,
            transform_eval=True),
        instance_postprocess_cfg=dict(max_per_image=100, focus_on_box=False)),
    init_cfg=None)

# dataset settings
data_root = '/data/work_folder/data/train_data/OFFICE_COCO/'

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    # dict(type='RandAugment', aug_space=color_space, aug_num=1),  # 会有问题
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False, backend='pillow'),
    dict(
        type='LoadPanopticAnnotations',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(    
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + '/annotations/panoptic_train.json',
        seg_prefix=data_root + '/annotations/panoptic_train/'),
    # dict(
    #     type='CocoMetric',
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     metric=['bbox', 'segm']),
    # dict(type='SemSegMetric', iou_metrics=['mIoU'])
]
test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 160000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

interval = 2000
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
