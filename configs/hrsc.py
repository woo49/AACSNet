_base_ = [
    'mmrotate::_base_/datasets/hrsc.py',
    'mmrotate::_base_/schedules/schedule_1x.py',
    'mmrotate::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.AACSNet.orientedformer'], allow_failed_imports=False)

num_stages = 2
num_proposals = 300
num_classes = 1  # HRSC only has 'ship' class
angle_version = 'le90'

# Override data_root (modify to your path)
data_root = 'data/HRSC2016/'

model = dict(
    type='OrientedDDQRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapperWithGN',
        kernel_size=1,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedAdaMixerDDQ',
        angle_version=angle_version,
        ddq_num_classes=num_classes,
        num_proposals=num_proposals,
        in_channels=256,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        dqs_cfg=dict(type='nms_rotated', iou_threshold=0.7, nms_pre=1000),
        offset=0.5,
        aafam_cfg=dict(
            num_angle_hypotheses=3,
            kernel_size=3,
            num_samples=9,
            enable_periodic_encoding=True,
            enable_rotation_sampling=True,
            enable_multi_angle_fusion=True,
            enable_angle_spatial_attention=True
        ),
        clsm_cfg=dict(
            embed_channels=256,
            context_channels=128,
            residual_ratio=0.5
        ),
        aux_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                                  box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)),
                alpha=1,
                beta=6)),
        main_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                                  box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)),
                alpha=1,
                beta=6))),
    roi_head=dict(
        type='OrientedAdaMixerDecoder',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=256,
        featmap_strides=[4, 8, 16, 32, 64],
        bbox_head=[
            dict(
                type='OrientedFormerDecoderLayer',
                num_classes=num_classes,
                angle_version=angle_version,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                num_cls_fcs=1,
                num_reg_fcs=1,
                content_dim=256,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1., 1., 1., 1., 1.),
                self_attn_cfg=dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0),
                o3d_attn_cfg=dict(
                    type='OrientedAttention',
                    n_points=32,
                    n_heads=64,
                    embed_dims=256,
                    reduction=4
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder')) for stage_idx in range(num_stages)
        ]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                        dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                    ]),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=36,
    val_interval=1,
    dynamic_intervals=None)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

# Override dataset config
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root))

val_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
test_evaluator = val_evaluator

# Runtime settings
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmrotate.RotLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
