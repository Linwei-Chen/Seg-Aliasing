dataset_type = 'CityscapesDataset'
data_root = '/home/ubuntu/2TB/dataset/'
# data_root = '/data3/chenlinwei/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (769, 769)
crop_size = (768, 768)
# crop_size = (129, 129)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
        # img_scale=(257, 129),
        # img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        # flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='AddNoisyImg', sigma=20,),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

BATCH_SIZE = 8
GPU = 1
data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cityscapes/leftImg8bit/train',
        ann_dir='cityscapes/gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cityscapes/leftImg8bit/val',
        ann_dir='cityscapes/gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='cityscapes/leftImg8bit/val',
        ann_dir='cityscapes/gtFine/val',
        pipeline=test_pipeline))

# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000)
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

ITER_SETTING = 80000
optimizer = dict(type='SGD', lr=0.01 / 8 * GPU * BATCH_SIZE, momentum=0.9, weight_decay=0.0005,
# optimizer = dict(type='SGD', lr=0.01,  momentum=0.9, weight_decay=0.0005,
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=2.),
            'att': dict(lr_mult=2.),
            }))
# optimizer = dict(type='SGD', lr=0.01 / 16 * GPU * BATCH_SIZE, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, 
                 min_lr=0.0001, ### 
                #  min_lr=0.0, ### 
                 by_epoch=False, 
                warmup='linear', warmup_iters=200)
# lr_config = dict(warmup='linear', warmup_iters=200) # polyrend
runner = dict(type='IterBasedRunner', max_iters=ITER_SETTING)
checkpoint_config = dict(by_epoch=False, interval=ITER_SETTING // 10, max_keep_ckpts=2)
evaluation = dict(interval=ITER_SETTING // 10, metric='mIoU', pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)

LOG_DIR = '/data3/chenlinwei/code/ResolutionDet/mmseg_exp/dct_log'
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        # type='ResNetV1c',
        # frozen_stages=4,
        type='ResNetV1cWithBlur',
        # type='NyResNet',
        # blur_type='adafreq',
        blur_type='adablur',
        # blur_type='blur',
        # blur_type='flc',
        # blur_type='none',
        # freq_thres=0.25 * 1.4,
        # blur_k=1,
        # log_aliasing_ratio=True,

        # with_cp=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UPerHead',
        # type='UPerHeadFreqMix',
        # upsampling_mode='bilinear',
        # k_lists=[[2, 4, 8], [2, 4, 8], [2, 4, 8], [2, 4, 8]],
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=(crop_size[0] % 2 == 1),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ]
            ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=(crop_size[0] % 2 == 1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole')
    # test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
    test_cfg=dict(mode='slide', crop_size=(768 + (crop_size[0] % 2), 768 + (crop_size[0] % 2)), stride=(512 + (crop_size[0] % 2), 512 + (crop_size[0] % 2)))
    )
work_dir = '/home/ubuntu/code/.'