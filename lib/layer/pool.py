from torch import nn

from lib.layer.conv import SamePaddingConv2d


class SamePaddingMaxPool2d(SamePaddingConv2d):
    def __init__(self, kernel_size, stride=1):
        super(SamePaddingMaxPool2d, self).__init__(1, 1, kernel_size, stride)

        self.pool = nn.MaxPool2d(kernel_size, stride)

        self.conv = self.pool
