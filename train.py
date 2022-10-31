import argparse
import math
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

from lib.block.efficientdet import Loss
from lib.model.efficientdet import EfficientDet
from lib.utility.config import Config
from lib.utility.dataset import CustomCocoDetection, CustomToTensor, CustomNormalize, CustomResize, CustomPad, \
    collate_fn
from lib.utility.device import get_device
from val import val


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', default='coco2017')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-w', '--weight')
    parser.add_argument('-l', '--logdir')
    parser.add_argument('-s', '--scout', default=10, type=int)
    parser.add_argument('-r', '--resolution', default=512, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config(Path(f'./project/{args.project}.yaml'))
    device = get_device()
    writer = SummaryWriter(args.logdir)

    args.batch_size *= torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainset = CustomCocoDetection(Path(config.trainset['root']), Path(config.trainset['annFile']),
                                   Compose([CustomToTensor(),
                                            CustomNormalize(config.mean, config.std),
                                            CustomResize(args.resolution),
                                            CustomPad(args.resolution)]))
    trainloader = DataLoader(trainset, args.batch_size, True,
                             num_workers=args.batch_size, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    valset = CustomCocoDetection(Path(config.valset['root']), Path(config.valset['annFile']),
                                 Compose([CustomToTensor(),
                                          CustomNormalize(config.mean, config.std),
                                          CustomResize(args.resolution),
                                          CustomPad(args.resolution)]))
    valloader = DataLoader(valset, args.batch_size, False,
                           num_workers=args.batch_size, collate_fn=collate_fn, pin_memory=True, drop_last=False)

    snapshot = torch.load(args.weight) if args.weight else None

    net = EfficientDet(len(config.categories), eval(config.anchors['scales']), eval(config.anchors['ratios']),
                       config.anchors['levels'], config.scale_feat, config.i_c, config.compound_coef, config.scale)
    if snapshot:
        net.load_state_dict(snapshot['net'])
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    criterion = Loss()
    optimizer = torch.optim.AdamW(net.parameters())
    if snapshot:
        optimizer.load_state_dict(snapshot['optim'])

    best_loss = snapshot['best_loss'] if snapshot else math.inf
    best_epoch = snapshot['best_epoch'] if snapshot else 0
    for epoch in count(snapshot['epoch'] + 1 if snapshot else 0):
        net.train()
        epoch_loss = []
        tq = tqdm(trainloader)
        for image, boxes, classes in tq:
            output = net(image.to(device))
            loss = criterion(*output, [b.to(device) for b in boxes], [c.to(device) for c in classes])

            tq.set_description(f'Epoch {epoch}, loss {[l.item() if hasattr(l, "item") else l for l in loss]}')

            loss = sum(loss)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        writer.add_scalar('train/loss', sum(epoch_loss), epoch)

        loss = val(net, valloader, criterion, device)
        writer.add_scalar('val/loss', loss, epoch)

        snapshot = {'net': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                    'optim': optimizer.state_dict(),
                    'best_loss': loss if loss < best_loss else best_loss,
                    'best_epoch': epoch if loss < best_loss else best_epoch,
                    'epoch': epoch}
        torch.save(snapshot, Path(writer.log_dir).joinpath('last.pth'))

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save(snapshot, Path(writer.log_dir).joinpath('best.pth'))

        if epoch - best_epoch >= args.scout:
            writer.add_text('val/conclusion', f'Best weight at epoch {best_epoch}.')
            break
