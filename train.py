import argparse
import logging
import math
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

from lib.block.efficientdet import Loss
from lib.model.efficientdet import EfficientDet
from lib.utility.config import Config
from lib.utility.device import get_device
from lib.utils import transforms
from lib.utils.data import collate_fn, getDataLoader
from val import val

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-s', '--size', default=512, type=int)
    parser.add_argument('-c', '--scout', default=10, type=int)
    parser.add_argument('-p', '--project', default='coco2017')
    parser.add_argument('-l', '--log_dir')
    parser.add_argument('-w', '--weight')
    return parser.parse_args()


if __name__ == '__main__':
    args = getArgs()
    project = Config(Path(f'project/{args.project}.yaml'))
    device = get_device()
    writer = SummaryWriter(args.log_dir)

    args.batch_size *= torch.cuda.device_count() if torch.cuda.is_available() else 1
    transf = Compose([transforms.EfficientToTensor(),
                      transforms.EfficientNormalize(project.mean, project.std),
                      transforms.EfficientResize(args.size),
                      transforms.EfficientPad(args.size)])
    train_loader = getDataLoader(project.trainset['root'], project.trainset['annFile'], transforms,
                                 batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = getDataLoader(project.valset['root'], project.valset['annFile'], transforms,
                               batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    snapshot = torch.load(args.weight) if args.weight else None

    net = EfficientDet(len(project.categories), eval(project.anchors['scales']), eval(project.anchors['ratios']),
                       project.anchors['levels'], project.scale_feat, project.i_c, project.compound_coef, project.scale)
    if snapshot:
        net.load_state_dict(snapshot['net'])
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    criterion = Loss(iou_loss_w=project.iou_loss_w)

    optim = torch.optim.AdamW(net.parameters())
    if snapshot:
        optim.load_state_dict(snapshot['optim'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=args.scout // 2, verbose=True)

    best_loss = snapshot['best_loss'] if snapshot else math.inf
    best_epoch = snapshot['best_epoch'] if snapshot else 0
    step = count(snapshot['step'] + 1 if snapshot else 0)
    for epoch in count(snapshot['epoch'] + 1 if snapshot else 0):
        net.train()
        losses = []
        s = 0
        progress = tqdm(train_loader)
        for images, boxes, classes in progress:
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            classes = [c.to(device) for c in classes]

            output = net(images)
            l_cla, l_reg, l_iou = criterion(*output, boxes, classes)

            loss = [l.item() if hasattr(l, 'item') else l for l in [l_cla, l_reg, l_iou]]
            progress.set_description(f'Epoch {epoch}, loss {loss}')

            s = next(step)
            if l_cla and torch.isfinite(l_cla):
                writer.add_scalar('train/classification_loss', loss[0], s)
            if l_reg and torch.isfinite(l_reg):
                writer.add_scalar('train/regression_loss', loss[1], s)
            if l_iou and torch.isfinite(l_iou):
                writer.add_scalar('train/iou_loss', loss[2], s)

            loss = sum([l_cla, l_reg, l_iou])
            if torch.isfinite(loss):
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())

        losses = sum(losses)
        writer.add_scalar('train/sum_loss', losses, epoch)
        scheduler.step(losses)

        loss = val(net, val_loader, criterion, device)
        writer.add_scalar('val/sum_loss', loss, epoch)

        snapshot = {'net': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                    'optim': optim.state_dict(),
                    'best_loss': loss if loss < best_loss else best_loss,
                    'best_epoch': epoch if loss < best_loss else best_epoch,
                    'step': s,
                    'epoch': epoch}
        torch.save(snapshot, Path(writer.log_dir).joinpath('last.pth'))

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            torch.save(snapshot, Path(writer.log_dir).joinpath('best.pth'))

        if epoch - best_epoch >= args.scout:
            writer.add_text('val/conclusion', f'Best weight at epoch {best_epoch}.')
            break
