# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES

import numpy as np

@PIPELINES.register_module()
class AddNoisyImg(object):
    def __init__(self, sigma=10.):
        self.sigma = sigma

    def __call__(self, results):
        print(f'Add noise: sigma = {self.sigma}')
        noise = np.random.normal(scale=self.sigma, size=results['img'].shape)
        for key in results.get('img_fields', ['img']):
            results['img'] = np.clip(results['img'].astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        return results
