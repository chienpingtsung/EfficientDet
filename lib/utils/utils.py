import argparse

import torch
import yaml


def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-s', '--scout', default=10, type=int)
    parser.add_argument('-i', '--size', default=512, type=int)
    parser.add_argument('-c', '--score_thr', default=0.05, type=float)
    parser.add_argument('-p', '--project', default='coco2017')
    parser.add_argument('-l', '--log_dir')
    parser.add_argument('-w', '--weight')

    args = parser.parse_args()

    with open(f'project/{args.project}.yaml') as file:
        project = yaml.safe_load(file.read())

        for k in project:
            if not hasattr(args, k):
                setattr(args, k, project[k])

    return args
