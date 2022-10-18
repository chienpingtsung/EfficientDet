import itertools

import numpy as np
import torch

from torch import nn


class Anchors(nn.Module):
    def __init__(self, anchor_scale, pyramid_levels, scales, ratios):
        super(Anchors, self).__init__()

        self.anchor_scale = anchor_scale
        self.strides = [2 ** x for x in pyramid_levels]
        self.scales = scales
        self.ratios = ratios

        self.last_shape = None
        self.last_anchors = {}

    def forward(self, x):
        *_, h, w = x.shape
        if (h, w) == self.last_shape and x.device in self.last_anchors:
            return self.last_anchors[x.device]
        if (h, w) != self.last_shape:
            self.last_shape = (h, w)

        dtype = x.dtype
        device = x.device

        anchors = []
        for stride in self.strides:
            assert h % stride == 0 and w % stride == 0

            anchors_level = []
            for scale, (ratio_w, ratio_h) in itertools.product(self.scales, self.ratios):
                anchor_size = self.anchor_scale * stride * scale
                anchor_size_w_half = anchor_size * ratio_w / 2
                anchor_size_h_half = anchor_size * ratio_h / 2

                x = np.arange(stride / 2, w, stride)
                y = np.arange(stride / 2, h, stride)
                x, y = np.meshgrid(x, y)
                x = x.reshape(-1)
                y = y.reshape(-1)

                boxes = np.vstack((x - anchor_size_w_half,
                                   y - anchor_size_h_half,
                                   x + anchor_size_w_half,
                                   y + anchor_size_h_half))
                boxes = boxes.transpose()
                anchors_level.append(np.expand_dims(boxes, 1))
            anchors_level = np.concatenate(anchors_level, 1)
            anchors.append(anchors_level.reshape((-1, 4)))
        anchors = np.vstack(anchors)

        anchors = torch.from_numpy(anchors).to(dtype=dtype, device=device)
        anchors = anchors.unsqueeze(0)
        self.last_anchors[device] = anchors

        return anchors
