import torch
from torch import nn
from torchvision import ops


class BoxDecoder(nn.Module):
    def forward(self, regression, anchors):
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2]
        anchor_h = anchors[..., 3]

        cx = regression[..., 0] * anchor_w + anchor_cx
        cy = regression[..., 1] * anchor_h + anchor_cy
        w = torch.exp(regression[..., 2]) * anchor_w / 2
        h = torch.exp(regression[..., 3]) * anchor_h / 2

        return torch.stack([cx - w, cy - h, cx + w, cy + h], dim=2)
