# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg_custom.models import build_segmentor
import time
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        # default=[2048, 1024],
        default=[64, 64],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    with torch.no_grad():
        # flops, params = get_model_complexity_info(model, input_shape, as_strings=False,)
        flops, params = get_model_complexity_info(model, input_shape, as_strings=True,)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    # exit()
    fake_input = torch.rand(1, 3, args.shape[-2], args.shape[-1]).cuda()
    time_list = []
    for _ in tqdm(range(1000)):
        t0 = time.perf_counter()
        _ = model(fake_input) 
        used_time = time.perf_counter() - t0
        time_list.append(used_time)
    print(sum(time_list) / len(time_list))

if __name__ == '__main__':
    main()
