import torch
from torch import nn

from lib.activation.swish import MemoryEfficientSwish
from lib.layer.conv import SamePaddingConv2d
from lib.layer.regularization import StochasticDepth


class MBConvBlock(nn.Module):
    def __init__(self, i_c, o_c, k_s, stride, bn_eps, bn_mom, expand_ratio, se_ratio, sd_ratio):
        super(MBConvBlock, self).__init__()

        self.swish = MemoryEfficientSwish()

        m_c = i_c * expand_ratio

        self.expand_conv = None
        self.expand_bn = None
        if expand_ratio != 1:
            self.expand_conv = SamePaddingConv2d(i_c, m_c, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(m_c, bn_eps, bn_mom)

        self.depthwise_conv = SamePaddingConv2d(m_c, m_c, k_s, stride, groups=m_c, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(m_c, bn_eps, bn_mom)

        self.se_pool = None
        self.se_sque = None
        self.se_exci = None
        if se_ratio:
            se_c = max(1, int(i_c * se_ratio))
            self.se_pool = nn.AdaptiveAvgPool2d(1)
            self.se_sque = SamePaddingConv2d(m_c, se_c, 1)
            self.se_exci = SamePaddingConv2d(se_c, m_c, 1)

        self.pointwise_conv = SamePaddingConv2d(m_c, o_c, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)

        self.sd = None
        if sd_ratio:
            self.sd = StochasticDepth(sd_ratio)

    def forward(self, x):
        x0 = x

        if self.expand_conv and self.expand_bn:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.swish(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.swish(x)

        if self.se_pool and self.se_sque and self.se_exci:
            se_x = self.se_pool(x)
            se_x = self.se_sque(se_x)
            se_x = self.swish(se_x)
            se_x = self.se_exci(se_x)
            x *= torch.sigmoid(se_x)

        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)

        if x0.shape == x.shape:
            if self.sd:
                x = self.sd(x)
            x = x + x0

        return x


class FusedMBConvBlock(MBConvBlock):
    def __init__(self, i_c, o_c, k_s, stride, bn_eps, bn_mom, expand_ratio, sd_ratio):
        super(FusedMBConvBlock, self).__init__(i_c, o_c, k_s, stride, bn_eps, bn_mom, 1, 0, sd_ratio)

        if expand_ratio != 1:
            m_c = i_c * expand_ratio
            self.depthwise_conv = SamePaddingConv2d(i_c, m_c, k_s, stride, bias=False)
            self.depthwise_bn = nn.BatchNorm2d(m_c, bn_eps, bn_mom)
            self.pointwise_conv = SamePaddingConv2d(m_c, o_c, 1, bias=False)
            self.pointwise_bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)
        else:
            self.depthwise_conv = SamePaddingConv2d(i_c, o_c, k_s, stride, bias=False)
            self.depthwise_bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)
            self.pointwise_conv = nn.Identity()
            self.pointwise_bn = nn.Identity()


class Stem(nn.Module):
    def __init__(self, i_c, o_c, k_s, stride, bn_eps, bn_mom):
        super(Stem, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.conv = SamePaddingConv2d(i_c, o_c, k_s, stride, bias=False)
        self.bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x


class Head(nn.Module):
    def __init__(self, i_c, o_c, bn_eps, bn_mom, dropout, num_classes):
        super(Head, self).__init__()

        self.swish = MemoryEfficientSwish()

        self.conv = SamePaddingConv2d(i_c, o_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(o_c, bn_eps, bn_mom)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(o_c, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)

        b, *_ = x.shape

        x = self.pool(x)
        x = x.view(b, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
