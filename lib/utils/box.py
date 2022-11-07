import torch
from torchvision import ops


class Box:
    @staticmethod
    def decode(anchors, adjustments):
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')

        anc_cx = anchors[..., 0]
        anc_cy = anchors[..., 1]
        anc_w = anchors[..., 2]
        anc_h = anchors[..., 3]

        cx = adjustments[..., 0] * anc_w + anc_cx
        cy = adjustments[..., 1] * anc_h + anc_cy
        w = torch.exp(adjustments[..., 2]) * anc_w
        h = torch.exp(adjustments[..., 3]) * anc_h

        return ops.box_convert(torch.stack([cx, cy, w, h], dim=-1), 'cxcywh', 'xyxy')

    @staticmethod
    def encode(anchors, boxes):
        anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
        boxes = ops.box_convert(boxes, 'xyxy', 'cxcywh')

        anc_cx = anchors[..., 0]
        anc_cy = anchors[..., 1]
        anc_w = anchors[..., 2]
        anc_h = anchors[..., 3]
        box_cx = boxes[..., 0]
        box_cy = boxes[..., 1]
        box_w = torch.clamp(boxes[..., 2], min=1)
        box_h = torch.clamp(boxes[..., 3], min=1)

        dx = (box_cx - anc_cx) / anc_w
        dy = (box_cy - anc_cy) / anc_h
        dw = torch.log(box_w / anc_w)
        dh = torch.log(box_h / anc_h)

        return torch.stack([dx, dy, dw, dh], dim=-1)
