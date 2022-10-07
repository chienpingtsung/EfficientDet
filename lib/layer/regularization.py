import torch
from torch import nn


class StochasticDepth(nn.Module):
    def __init__(self, p):
        super(StochasticDepth, self).__init__()

        self.keep_p = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        b, *_ = x.shape

        random_tensor = torch.rand((b, 1, 1, 1), dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor + self.keep_p)

        return x / self.keep_p * binary_tensor
