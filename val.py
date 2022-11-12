import json
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from torchvision import ops, utils
from torchvision.transforms import Compose
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from lib.utils import transforms
from lib.utils.box import Box
from lib.utils.data import getDataLoader, collate_fn
from lib.utils.utils import getArgs, getDevice, getNet


def val(net, dataloader, device, args, criterion=None, coco=None, id_map_reverse=None, visualization=None):
    results = []
    if coco and id_map_reverse:
        coco = Path(coco)
        coco.parent.mkdir(parents=True, exist_ok=True)

    if visualization:
        visualization = Path(visualization)
        visualization.mkdir(parents=True, exist_ok=True)

    net.eval()

    val_losses = []

    for image, boxes, cats, meta in dataloader:
        with torch.no_grad():
            cla, reg, anc = net(image.to(device))

            if criterion:
                loss = sum(criterion(cla, reg, anc, [b.to(device) for b in boxes], [c.to(device) for c in cats]))
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

            if (coco and id_map_reverse) or visualization:
                for c, r, (image, image_id, file_name) in zip(torch.sigmoid(cla), reg, meta):
                    w, h = image.size
                    scale = args.size / max(h, w)

                    scores, categories = torch.max(c, dim=-1)

                    indices = scores > args.score_thr

                    a = anc[indices]
                    r = r[indices]
                    boxes = Box.decode(a, r)
                    scores = scores[indices]
                    categories = categories[indices]

                    indices = ops.batched_nms(boxes, scores, categories, args.iou_threshold)

                    boxes = ops.clip_boxes_to_image(boxes[indices] / scale, [h, w])
                    scores = scores[indices]
                    categories = categories[indices]

                    if coco and id_map_reverse:
                        for b, s, c in zip(ops.box_convert(boxes, 'xyxy', 'xywh'), scores, categories):
                            results.append({'image_id': image_id,
                                            'bbox': b.tolist(),
                                            'score': s.item(),
                                            'category_id': id_map_reverse[c.item()]})

                    if visualization:
                        labels = [f'{args.categories[c.item()]}@{s.item():.2f}' for s, c in zip(scores, categories)]
                        vis = utils.draw_bounding_boxes(pil_to_tensor(image), boxes, labels, width=3)
                        utils.save_image(vis.to(torch.float) / 255, visualization.joinpath(file_name))

    if coco and id_map_reverse:
        with open(coco, 'w') as file:
            json.dump(results, file, indent=4)

    return sum(val_losses)


def eval(cocoGt, cocoDt):
    cocoDt = cocoGt.loadRes(cocoDt)

    cocoeval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()


if __name__ == '__main__':
    args = getArgs()
    device = getDevice()

    trans = Compose([transforms.EfficientToTensor(),
                     transforms.EfficientNormalize(args.mean, args.std),
                     transforms.EfficientResize(args.size),
                     transforms.EfficientPad(args.size)])
    dataloader = getDataLoader(args.valset['root'], args.valset['annFile'], trans,
                               batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    snapshot = torch.load(args.weight, map_location=device)

    net = getNet(args, device, parallel=torch.cuda.device_count() > 1, snapshot=snapshot['net'])

    progress = tqdm(dataloader)
    progress.set_description('Testing')

    val(net, progress, device, args,
        coco='val/coco.json', id_map_reverse=dataloader.dataset.id_map_reverse,
        visualization='val/visualization')

    eval(dataloader.dataset.coco, 'val/coco.json')
