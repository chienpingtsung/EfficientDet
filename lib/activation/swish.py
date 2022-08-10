import torch
from torch import nn

Swish = nn.SiLU


class MemoryEfficientSwishImplement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwishImplement.apply(x)
