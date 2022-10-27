import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CustomCocoDetection, self).__init__(root, annFile)
        self.custom_transforms = transforms
        self.id_map = {}
        for idx, cat in enumerate(self.coco.dataset['categories']):
            self.id_map[cat['id']] = idx

    def __getitem__(self, item):
        sample = super(CustomCocoDetection, self).__getitem__(item)
        if self.custom_transforms:
            sample = self.custom_transforms(sample)
        image, boxes, classes = sample
        classes = [self.id_map[x] for x in classes]
        return image, boxes, torch.from_numpy(np.asarray(classes))


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

        return image, torch.from_numpy(boxes), classes


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
        return functional.pad(image, [0, 0, right, bottom], self.fill, self.padding_mode), boxes, classes


def collate_fn(data):
    image = []
    boxes = []
    classes = []
    for i, b, c in data:
        image.append(i)
        boxes.append(b)
        classes.append(c)
    image = torch.stack(image)
    return image, boxes, classes
