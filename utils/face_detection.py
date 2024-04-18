import cv2
import torch
from itertools import product as product
import numpy as np
import math
from math import ceil

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class GeneartePriorBoxes(object):
    '''
    both for fpn and single layer, single layer need to test
    return (np.array) [num_anchros, 4] [x0, y0, x1, y1]
    '''
    def __init__(self, scale_list=[1.], \
                 aspect_ratio_list=[1.0], \
                 stride_list=[4,8,16,32,64,128], \
                 anchor_size_list=[16,32,64,128,256,512]):
        self.scale_list = scale_list
        self.aspect_ratio_list = aspect_ratio_list
        self.stride_list = stride_list
        self.anchor_size_list = anchor_size_list
    
    @staticmethod
    def normalize_anchor(anchors):
        '''
        from  [c_x, cy, w, h] to [x0, x1, y0, y1] 
        '''
        return np.concatenate((anchors[:, :2] - (anchors[:, 2:] - 1) / 2,
                               anchors[:, :2] + (anchors[:, 2:] - 1) / 2), axis=1) 

    def __call__(self, img_height, img_width):
        final_anchor_list = []

        for idx, stride in enumerate(self.stride_list):
            anchor_list = []
            cur_img_height = img_height
            cur_img_width = img_width
            tmp_stride = stride 

            while tmp_stride != 1:
                tmp_stride = tmp_stride // 2
                cur_img_height = (cur_img_height + 1) // 2
                cur_img_width = (cur_img_width + 1) // 2

            for i in range(cur_img_height):
                for j in range(cur_img_width):
                    for scale in self.scale_list:
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        side_x = self.anchor_size_list[idx] * scale
                        side_y = self.anchor_size_list[idx] * scale
                        for ratio in self.aspect_ratio_list:
                            anchor_list.append([cx, cy, side_x / math.sqrt(ratio), side_y * math.sqrt(ratio)])

            final_anchor_list.append(anchor_list)
        final_anchor_arr = np.concatenate(final_anchor_list, axis=0)
        normalized_anchor_arr = self.normalize_anchor(final_anchor_arr).astype('float32')

        return normalized_anchor_arr