import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import ops, utils
from torchvision.transforms import Compose, ToPILImage, PILToTensor
from tqdm import tqdm

from lib.model.efficientdet import EfficientDet
from lib.utility.box import BoxDecoder
from lib.utility.config import Config
from lib.utility.device import get_device
from lib.utils import transforms
from lib.utils.data import EfficientCocoDetection
from lib.utils.data import collate_fn

logger = logging.getLogger()


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-s', '--size', default=512, type=int)
    parser.add_argument('-c', '--scout', default=10, type=int)
    parser.add_argument('-p', '--project', default='coco2017')
    parser.add_argument('-l', '--log_dir')
    parser.add_argument('-w', '--weight')
    return parser.parse_args()


def getDataLoader(root, annFile, transforms, batch_size, shuffle, collate_fn, drop_last):
    dataset = EfficientCocoDetection(Path(root), Path(annFile), transforms)
    num_workers = min(batch_size, os.cpu_count())
    return DataLoader(dataset, batch_size, shuffle,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=drop_last)


def val(net, dataloader, criterion, device):
    net.eval()
    epoch_loss = []
    progress = tqdm(dataloader)
    for image, boxes, classes in progress:
        with torch.no_grad():
            output = net(image.to(device))
            loss = sum(criterion(*output, [b.to(device) for b in boxes], [c.to(device) for c in classes]))

            progress.set_description(f'Test loss {loss.item()}')

            if torch.isfinite(loss):
                epoch_loss.append(loss.item())
    return sum(epoch_loss)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = getArgs()
    project = Config(f'project/{args.project}.yaml')
    device = get_device()

    args.batch_size = 1
    transf = Compose([transforms.EfficientToTensor(),
                      transforms.EfficientNormalize(project.mean, project.std),
                      transforms.EfficientResize(args.size),
                      transforms.EfficientPad(args.size)])
    val_loader = getDataLoader(project.valset['root'], project.valset['annFile'], transforms,
                               batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    snapshot = torch.load(args.weight, map_location=torch.device('cpu'))

    net = EfficientDet(len(project.categories), eval(project.anchors['scales']), eval(project.anchors['ratios']),
                       project.anchors['levels'], project.scale_feat, project.i_c, project.compound_coef, project.scale)
    net.load_state_dict(snapshot['net'])
    net.to(device)

    decoder = BoxDecoder()
    toPILimage = ToPILImage()
    toTensor = PILToTensor()

    net.eval()
    progress = tqdm(val_loader)
    for ind, (image, boxes, classes, orgi_image) in enumerate(progress):
        with torch.no_grad():
            cla, reg, anc = net(image.to(device))
            cla = torch.sigmoid(cla)

            progress.set_description(f'Validating image {ind}')

            boxes = decoder(reg, anc)

            scores, categories = torch.max(cla, dim=2)

            indices = scores > 0.2
            scores = scores[indices]
            print(indices, scores.shape)
            categories = categories[indices]
            boxes = boxes[indices]

            indices = ops.batched_nms(boxes, scores, categories, 0.2)
            boxes = boxes[indices]
            categories = categories[indices]

            print(boxes, categories)
            w, h = orgi_image[0].size
            scale = args.size / max(h, w)
            boxes = boxes / scale

            boxes = ops.clip_boxes_to_image(boxes, [h, w])
            output = utils.draw_bounding_boxes(toTensor(orgi_image[0]), boxes,
                                               [project.categories[c.item()] for c in categories])

            output = toPILimage(output.to(torch.float) / 255)
            output.save(f'output/{ind}.png')
