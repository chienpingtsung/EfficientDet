import itertools

import numpy as np
import torch

from torch import nn
from torchvision import ops

from lib.activation.swish import MemoryEfficientSwish
from lib.layer.conv import SeparableConv2d, SamePaddingConv2d
from lib.layer.pool import SamePaddingMaxPool2d
from lib.utils.box import Box


class Anchors(nn.Module):
    def __init__(self, anchor_scale, levels, scales, ratios):
        super(Anchors, self).__init__()

        self.anchor_scale = anchor_scale
        self.strides = [2 ** x for x in levels]
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

        anchors = torch.from_numpy(anchors).to(device=device, dtype=dtype)
        self.last_anchors[device] = anchors

        return anchors


class SeparableConv2dBN(nn.Module):
    def __init__(self, i_c, o_c, k_s, bn_eps, bn_mom):
        super(SeparableConv2dBN, self).__init__()

        self.conv = SeparableConv2d(i_c, o_c, k_s, bias=False)
        self.bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SeparableConv2dBNSwish(SeparableConv2dBN):
    def __init__(self, *args, **kwargs):
        super(SeparableConv2dBNSwish, self).__init__(*args, **kwargs)

        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = super(SeparableConv2dBNSwish, self).forward(x)
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


class BiFPN(nn.Module):
    def __init__(self, c, bn_eps, bn_mom, pyramid_levels, attention, epsilon=1e-4):
        super(BiFPN, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.pyramid_levels = pyramid_levels
        self.attention = attention
        self.epsilon = epsilon

        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = SamePaddingMaxPool2d(3, 2)

        self.conv_down = nn.ModuleList(SeparableConv2dBN(c, c, 3, bn_eps, bn_mom)
                                       for _ in range(self.pyramid_levels - 1))
        self.conv_up = nn.ModuleList(SeparableConv2dBN(c, c, 3, bn_eps, bn_mom)
                                     for _ in range(self.pyramid_levels - 2))
        self.conv_sp = SeparableConv2dBN(c, c, 3, bn_eps, bn_mom)

        self.conv_down_w = torch.ones(self.pyramid_levels - 1, 2)
        self.conv_up_w = torch.ones(self.pyramid_levels - 2, 3)
        self.conv_sp_w = torch.ones(2)

        if self.attention:
            self.relu = nn.ReLU()

            self.conv_down_w = nn.Parameter(self.conv_down_w)
            self.conv_up_w = nn.Parameter(self.conv_up_w)
            self.conv_sp_w = nn.Parameter(self.conv_sp_w)

    def forward(self, x):
        *_, last = x
        down_features = [last]
        for i, conv, w in zip(range(self.pyramid_levels - 2, -1, -1),
                              self.conv_down,
                              self.conv_down_w):
            if self.attention:
                w = self.relu(w)
                w = w / (torch.sum(w) + self.epsilon)
            w1, w2 = w
            last = conv(self.swish(w1 * x[i] + w2 * self.upsample(last)))
            down_features.append(last)

        *_, last = down_features
        up_features = [last]
        for i, j, conv, w in zip(range(1, self.pyramid_levels - 1),
                                 range(self.pyramid_levels - 2, 0, -1),
                                 self.conv_up,
                                 self.conv_up_w):
            if self.attention:
                w = self.relu(w)
                w = w / (torch.sum(w) + self.epsilon)
            w1, w2, w3 = w
            last = conv(self.swish(w1 * x[i] + w2 * down_features[j] + w3 * self.downsample(last)))
            up_features.append(last)

        *_, x_last = x
        w = self.conv_sp_w
        if self.attention:
            w = self.relu(w)
            w = w / (torch.sum(w) + self.epsilon)
        w1, w2 = w
        up_features.append(self.conv_sp(self.swish(w1 * x_last + w2 * self.downsample(last))))

        return up_features


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, cla_loss_w=1, beta=1 / 9, reg_loss_w=50, iou_loss_w=None):
        super(Loss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.cla_loss_w = cla_loss_w

        self.smoothl1 = nn.SmoothL1Loss(reduction='mean', beta=beta)
        self.reg_loss_w = reg_loss_w

        self.iou_loss_w = iou_loss_w

    def forward(self, classification, regression, anchors, box_annotations, cla_annotations):
        classification_losses = []
        regression_losses = []
        iou_losses = []
        for cla, reg, box_ann, cla_ann in zip(classification, regression, box_annotations, cla_annotations):
            target = torch.zeros_like(cla)
            ignore = torch.zeros_like(cla).to(torch.bool)
            num_positive = 1
            if len(box_ann) and len(cla_ann):
                iou = ops.box_iou(anchors, box_ann)
                iou_max, iou_argmax = torch.max(iou, dim=1)

                negative = torch.lt(iou_max, 0.4)
                positive = torch.ge(iou_max, 0.5)
                ignore[torch.logical_not(torch.logical_or(negative, positive))] = True

                num_positive = torch.clamp(torch.sum(positive), min=1)

                box_ann, cla_ann = box_ann[iou_argmax], cla_ann[iou_argmax]

                target[positive, cla_ann[positive]] = 1

                box_ann = box_ann[positive]
                pos_reg = reg[positive]
                pos_anc = anchors[positive]

                box_enco = Box.encode(pos_anc, box_ann)
                loss = self.smoothl1(pos_reg, box_enco)
                if torch.isfinite(loss):
                    regression_losses.append(loss)

                if self.iou_loss_w:
                    box_deco = Box.decode(pos_anc, pos_reg)
                    loss = ops.complete_box_iou_loss(box_deco, box_ann, 'mean')
                    if torch.isfinite(loss):
                        iou_losses.append(loss)
            classification_loss = ops.sigmoid_focal_loss(cla, target, self.alpha, self.gamma)
            zeros = torch.zeros_like(classification_loss)
            classification_losses.append(torch.sum(torch.where(ignore, zeros, classification_loss)) / num_positive)
        return (self.cla_loss_w * torch.mean(torch.stack(classification_losses)),
                self.reg_loss_w * torch.mean(torch.stack(regression_losses)) if regression_losses else 0,
                self.iou_loss_w * torch.mean(torch.stack(iou_losses)) if iou_losses else 0)
