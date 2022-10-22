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

        return torch.stack([cx - w, cy - h, cx + w, cy + h], dim=1 if regression.shape == anchors.shape else 2)


class BoxEncoder(nn.Module):
    def forward(self, box_ann, anchors):
        box_ann = ops.box_convert(box_ann, 'xyxy', 'cxcywh')
        ann_cx = box_ann[..., 0]
        ann_cy = box_ann[..., 1]
        ann_w = torch.clamp(box_ann[..., 2], min=1)
        ann_h = torch.clamp(box_ann[..., 3], min=1)
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
        anchor_cx = anchors[..., 0]
        anchor_cy = anchors[..., 1]
        anchor_w = anchors[..., 2]
        anchor_h = anchors[..., 3]

        dx = (ann_cx - anchor_cx) / anchor_w
        dy = (ann_cy - anchor_cy) / anchor_h
        dw = torch.log(ann_w / anchor_w)
        dh = torch.log(ann_h / anchor_h)

        return torch.stack([dx, dy, dw, dh], dim=1)
