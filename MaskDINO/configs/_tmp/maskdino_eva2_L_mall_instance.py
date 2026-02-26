_base_ = [
    '_base_/datasets/lsj_mall-instance.py',
    '_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.MaskDINO.maskdino', 
             'projects.VideoKNet.models', 
             'projects.VideoKNet.datasets'], allow_failed_imports=False)

load_from = 'work_dirs/maskdino_eva2_L_coco_panoptic/iter_368750.pth'

image_size = (1024, 1024)
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

num_things_classes = 50
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='MaskDINO',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='EVA2',
        img_size=1024, 
        patch_size=16, 
        in_chans=3,
        embed_dim=1024, 
        depth=24,
        num_heads=16, 
        mlp_ratio=4*2/3,      # GLU default
        out_indices=[7, 11, 15, 23],
        
        qkv_bias=True, 
        drop_path_rate=0.2, 
        
        init_values=None, 
        use_checkpoint=False, 

        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,

        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,

        subln=True,
        xattn=True,
        naiveswiglu=True,

        pretrained=None,
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
            transformer_in_features=['res2', 'res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=4,
            total_num_feature_levels=5,
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
            total_num_feature_levels=5,
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
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        panoptic_postprocess_cfg=dict(
            object_mask_thr=0.25,  # 0.8 for MaskFormer
            iou_thr=0.8,
            filter_low_score=True,  # it will filter mask area where score is less than 0.5.
            panoptic_temperature=0.06,
            transform_eval=True),
        instance_postprocess_cfg=dict(max_per_image=100, focus_on_box=False)),
    init_cfg=None)

train_dataloader = dict(    
    batch_size=1,
    num_workers=2)


