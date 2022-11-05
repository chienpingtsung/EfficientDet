import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional


class EfficientToTensor(transforms.ToTensor):
    def __call__(self, sample):
        image, boxes, cats = sample

        image = super(EfficientToTensor, self).__call__(image)
        boxes = torch.from_numpy(np.asarray(boxes))
        cats = torch.from_numpy(np.asarray(cats))

        return image, boxes, cats


class EfficientNormalize(transforms.Normalize):
    def forward(self, sample):
        image, boxes, cats = sample
        return super(EfficientNormalize, self).forward(image), boxes, cats


class EfficientResize(transforms.Resize):
    def forward(self, sample):
        image, boxes, cats = sample

        *_, h, w = image.shape
        scale = self.size / max(h, w)
        h, w = int(h * scale), int(w * scale)

        image = functional.resize(image, [h, w], self.interpolation, self.max_size, self.antialias)

        return image, boxes * scale, cats


class EfficientPad(transforms.Pad):
    def forward(self, sample):
        image, boxes, cats = sample

        *_, h, w = image.shape
        right = self.padding - w
        bottom = self.padding - h

        return functional.pad(image, [0, 0, right, bottom], self.fill, self.padding_mode), boxes, cats


def collate_fn(data):
    image = []
    boxes = []
    cats = []
    meta = []
    for i, b, c, m in data:
        image.append(i)
        boxes.append(b)
        cats.append(c)
        meta.append(m)

    return torch.stack(image), boxes, cats, meta
