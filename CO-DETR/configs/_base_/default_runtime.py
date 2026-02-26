default_scope = 'mmdet'

interval = 10000

# max_iters = 40000
# train_cfg = dict(
#     type='IterBasedTrainLoop',
#     max_iters=max_iters,
#     val_interval=interval)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         T_max=38500,
#         begin=1500,
#         by_epoch=False,
#         end=max_iters,
#         type='CosineAnnealingLR'
#     )
# ]

# # optimizer
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         weight_decay=0.05,
#         eps=1e-8,
#         betas=(0.9, 0.999)),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#         },
#         norm_decay_mult=0.0),
#     clip_grad=dict(max_norm=0.01, norm_type=2))


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        interval=interval,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

log_level = 'INFO'
load_from = None
resume = False



