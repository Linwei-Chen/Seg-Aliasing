_base_ = './upernet_r50_512x512_40k_voc12aug.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(
        depth=101,
        # type='ResNetV1c',
        # type='ResNetV1cWithBlur',
        type='NyResNet',
        # blur_type='adafreq',
        # blur_type='blur',
        blur_type='flc',
        freq_thres=0.25 * 1.4,
        # blur_k=7,
        with_cp=True,
        # use_checkpoing=True,
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
evaluation = dict(save_best='mIoU', pre_eval='True')