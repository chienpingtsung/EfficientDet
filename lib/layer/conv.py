from torch import nn
from torch.nn import functional


class SamePaddingConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 device=None,
                 dtype=None):
        super(SamePaddingConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              device=device,
                              dtype=dtype)

        self.k = kernel_size + (kernel_size - 1) * (dilation - 1)
        self.s = stride

    def forward(self, x):
        *_, h, w = x.shape

        r_h = h % self.s
        p_h = self.k - (r_h if r_h else self.s)
        r_w = w % self.s
        p_w = self.k - (r_w if r_w else self.s)

        left = p_w // 2
        right = p_w - left
        top = p_h // 2
        bottom = p_h - top

        return self.conv(functional.pad(x, (left, right, top, bottom)))


class SeparableConv2d(nn.Module):
    def __init__(self, i_c, o_c, k_s, bias=True):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = SamePaddingConv2d(i_c, i_c, k_s, groups=i_c, bias=False)
        self.pointwise_conv = SamePaddingConv2d(i_c, o_c, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
