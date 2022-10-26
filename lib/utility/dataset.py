import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional


class CustomToTensor(transforms.ToTensor):
    def __call__(self, sample):
        image, target = sample

        image = super(CustomToTensor, self).__call__(image)

        boxes = []
        classes = []
        for t in target:
            boxes.append(t['bbox'])
            classes.append(t['category_id'])
        boxes = np.asarray(boxes)
        classes = np.asarray(classes)

        return image, torch.from_numpy(boxes), torch.from_numpy(classes)


class CustomNormalize(transforms.Normalize):
    def forward(self, sample):
        image, boxes, classes = sample
        return super(CustomNormalize, self).forward(image), boxes, classes


class CustomResize(transforms.Resize):
    def forward(self, sample):
        image, boxes, classes = sample
        *_, h, w = image.shape
        scale = self.size / max(h, w)
        h, w = int(h * scale), int(w * scale)
        image = functional.resize(image, [h, w], self.interpolation, self.max_size, self.antialias)
        return image, boxes * scale, classes


class CustomPad(transforms.Pad):
    def forward(self, sample):
        image, boxes, classes = sample
        *_, h, w = image.shape
        right = self.padding - w
        bottom = self.padding - h
        return functional.pad(image, [0, 0, right, bottom], self.fill, self.padding_mode)
