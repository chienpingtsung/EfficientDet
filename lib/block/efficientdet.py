import itertools

import numpy as np
import torch

from torch import nn

from lib.activation.swish import MemoryEfficientSwish
from lib.layer.conv import SeparableConv2d, SamePaddingConv2d
from lib.layer.pool import SamePaddingMaxPool2d


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


class SeparableConv2dBNSwish(nn.Module):
    def __init__(self, i_c, o_c, k_s, bn_eps, bn_mom):
        super(SeparableConv2dBNSwish, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.conv = SeparableConv2d(i_c, o_c, k_s, bias=False)
        self.bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x


class Classifier(nn.Module):
    def __init__(self, i_c, num_anchors, num_classes, num_layers, pyramid_levels, bn_eps, bn_mom):
        super(Classifier, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.num_classes = num_classes

        self.conv_list = nn.ModuleList(SeparableConv2d(i_c, i_c, 3, bias=False)
                                       for _ in range(num_layers))
        self.bn_list = nn.ModuleList(nn.ModuleList(nn.BatchNorm2d(i_c, bn_eps, bn_mom)
                                                   for _ in range(num_layers))
                                     for _ in range(pyramid_levels))
        self.head = SeparableConv2d(i_c, num_anchors * num_classes, 3, bias=True)

    def forward(self, x):
        boxes_cla = []
        for feat, bn_list in zip(x, self.bn_list):
            for conv, bn in zip(self.conv_list, bn_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.head(feat)

            b, _, h, w = feat.shape
            feat = feat.view(b, -1, self.num_classes, h, w)
            feat = feat.permute(0, 3, 4, 1, 2)
            feat = feat.reshape(b, -1, self.num_classes)
            boxes_cla.append(feat)
        boxes_cla = torch.cat(boxes_cla, 1)
        boxes_cla = torch.sigmoid(boxes_cla)
        return boxes_cla


class Regressor(nn.Module):
    def __init__(self, i_c, num_anchors, num_layers, pyramid_levels, bn_eps, bn_mom):
        super(Regressor, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.conv_list = nn.ModuleList(SeparableConv2d(i_c, i_c, 3, bias=False)
                                       for _ in range(num_layers))
        self.bn_list = nn.ModuleList(nn.ModuleList(nn.BatchNorm2d(i_c, bn_eps, bn_mom)
                                                   for _ in range(num_layers))
                                     for _ in range(pyramid_levels))
        self.head = SeparableConv2d(i_c, num_anchors * 4, 3, bias=True)

    def forward(self, x):
        boxes_reg = []
        for feat, bn_list in zip(x, self.bn_list):
            for conv, bn in zip(self.conv_list, bn_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.head(feat)

            b, _, h, w = feat.shape
            feat = feat.view(b, -1, 4, h, w)
            feat = feat.permute(0, 3, 4, 1, 2)
            feat = feat.reshape(b, -1, 4)
            boxes_reg.append(feat)
        boxes_reg = torch.cat(boxes_reg, 1)
        boxes_reg = torch.sigmoid(boxes_reg)
        return boxes_reg


class FPNStem(nn.Module):
    def __init__(self, p_i_c, o_c, pyramid_levels, bn_eps, bn_mom):
        super(FPNStem, self).__init__()

        self.down_times = pyramid_levels - len(p_i_c)

        self.p_down_channel = nn.ModuleList()
        for i_c in p_i_c:
            self.p_down_channel.append(nn.Sequential(SamePaddingConv2d(i_c, o_c, 1, bias=False),
                                                     nn.BatchNorm2d(o_c, bn_eps, bn_mom)))

        if 0 < self.down_times:
            *_, i_c = p_i_c
            self.down_channel = nn.Sequential(SamePaddingConv2d(i_c, o_c, 1, bias=False),
                                              nn.BatchNorm2d(o_c, bn_eps, bn_mom))
            self.down_sample = SamePaddingMaxPool2d(3, 2)

    def forward(self, x):
        scale_features = []

        for feat, p_down_channel in zip(x, self.p_down_channel):
            scale_features.append(p_down_channel(feat))

        if 0 < self.down_times:
            *_, feat = x
            feat = self.down_channel(feat)
            for _ in range(self.down_times):
                feat = self.down_sample(feat)
                scale_features.append(feat)

        return scale_features
