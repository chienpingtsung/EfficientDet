import argparse

import torch
import yaml
from torch import nn

from lib.model.efficientdet import EfficientDet


def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-s', '--scout', default=10, type=int)
    parser.add_argument('-i', '--size', default=512, type=int)
    parser.add_argument('-p', '--project', default='coco2017')
    parser.add_argument('-l', '--log_dir')
    parser.add_argument('-w', '--weight')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--score_thr', default=0.05, type=float)

    args = parser.parse_args()

    with open(f'project/{args.project}.yaml') as file:
        project = yaml.safe_load(file.read())

        for k in project:
            if not hasattr(args, k):
                setattr(args, k, project[k])

    return args


def getNet(args, device, parallel=False, snapshot=None):
    net = EfficientDet(len(args.categories), eval(args.anchors['scales']), eval(args.anchors['ratios']),
                       args.anchors['levels'], args.scale_feat, args.i_c, args.compound_coef, args.scale)

    if snapshot:
        net.load_state_dict(snapshot)

    if parallel:
        net = nn.DataParallel(net)

    return net.to(device)
