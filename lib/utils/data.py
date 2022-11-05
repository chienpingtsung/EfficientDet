import torch
from torchvision.datasets import CocoDetection


class EfficientCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(EfficientCocoDetection, self).__init__(root, annFile)

        self.sub_transforms = transforms

        self.id_map = {}
        self.id_map_reverse = {}
        for ind, cat in enumerate(self.coco.dataset['categories']):
            self.id_map[cat['id']] = ind
            self.id_map_reverse[ind] = cat['id']

    def __getitem__(self, item):
        image, target = super(EfficientCocoDetection, self).__getitem__(item)
        image_id = self.ids[item]
        file_name = self.coco.imgs[image_id]['file_name']

        boxes = []
        cats = []
        for t in target:
            boxes.append(t['bbox'])
            cats.append(self.id_map[t['category_id']])

        sample = image, boxes, cats
        if self.sub_transforms:
            sample = self.sub_transforms(sample)

        return *sample, (image, image_id, file_name)


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
