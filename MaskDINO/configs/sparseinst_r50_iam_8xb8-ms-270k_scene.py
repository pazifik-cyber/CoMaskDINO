_base_ = [
    '_base_/datasets/lsj-scene-instance.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['projects.SparseInst.sparseinst',
             'projects.MaskDINO.maskdino', 
             'projects.VideoKNet.models', 
             'projects.VideoKNet.datasets'], allow_failed_imports=False)

num_classes = 134
load_from = 'ckpts/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth'
model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[512, 1024, 2048],
        out_channels=256),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2,
        num_classes=num_classes,
        ins_dim=256,
        ins_conv=4,
        mask_dim=256,
        mask_conv=4,
        kernel_dim=128,
        scale_factor=2.0,
        output_iam=False,
        num_masks=100),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=num_classes,
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45))

train_dataloader = dict(
    batch_size=8,
    num_workers=16)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict( type='AdamW', lr=0.00005, weight_decay=0.05))
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=270000,
    val_interval=10000)
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=270000,
        by_epoch=False,
        milestones=[210000, 250000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64, enable=True)
