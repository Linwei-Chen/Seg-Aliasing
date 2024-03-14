_base_ = './upernet_r50_512x512_40k_voc12aug.py'
model = dict(
    pretrained=None, 
    # pretrained='open-mmlab://resnet50_v1c', 
    backbone=dict(
        depth=50,
        # type='ResNetV1c',
        # type='ResNetV1cWithBlur',
        # type='NyResNetFreezePretrain',
        # frozen_stages=4,
        type='NyResNet',
        # type='ResNetFreqMix',
        # blur_type='adafreq',
        # blur_type='blur',
        blur_type='flc',
        freq_thres=0.25 * 1.4,
        # blur_k=7,
        # with_cp=True,
        # use_checkpoing=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet50_v1c',
            # prefix='backbone.'
            )
        ),
    decode_head=dict(
        type='UPerHead',
        channels=128,)
)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
)
checkpoint_config = dict(max_keep_ckpts=2)
optimizer = dict(
    paramwise_cfg = dict(
        custom_keys={
            # 'FPNDyHPAlign': dict(lr_mult=2.), 
            # 'FPNFADyHPAlign': dict(lr_mult=2.), 
            # 'FaPNDyHPAlign': dict(lr_mult=2.), 
            'head': dict(lr_mult=2.),
            'att': dict(lr_mult=2.),
            # 'comp_conv': dict(lr_mult=2.),
            }))
evaluation = dict(save_best='mIoU', pre_eval='True')