import torch
from torchvision import ops, utils
from torchvision.transforms import PILToTensor, Compose
from tqdm import tqdm

from lib.model.efficientdet import EfficientDet
from lib.utils import transforms
from lib.utils.box import Box
from lib.utils.data import getDataLoader, collate_fn
from lib.utils.utils import getArgs, getDevice


def val(net, dataloader, device, args, criterion=None, coco=None, visualization=None):
    piltotensor = PILToTensor()

    net.eval()

    val_losses = []

    for image, boxes, cats, meta in dataloader:
        with torch.no_grad():
            cla, reg, anc = net(image.to(device))

            if criterion:
                loss = sum(criterion(cla, reg, anc, [b.to(device) for b in boxes], [c.to(device) for c in cats]))
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

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

                labels = [f'{args.categories[c.item()]}@{s.item()}' for s, c in zip(scores, categories)]
                vis = utils.draw_bounding_boxes(piltotensor(image), boxes, labels, width=3)
                utils.save_image(vis.to(torch.float) / 255, f'output/{file_name}')

    return sum(val_losses)


if __name__ == '__main__':
    args = getArgs()
    device = getDevice()

    args.batch_size = 1
    transf = Compose([transforms.EfficientToTensor(),
                      transforms.EfficientNormalize(args.mean, args.std),
                      transforms.EfficientResize(args.size),
                      transforms.EfficientPad(args.size)])
    val_loader = getDataLoader(args.valset['root'], args.valset['annFile'], transf,
                               batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    snapshot = torch.load(args.weight, map_location=torch.device('cpu'))

    net = EfficientDet(len(args.categories), eval(args.anchors['scales']), eval(args.anchors['ratios']),
                       args.anchors['levels'], args.scale_feat, args.i_c, args.compound_coef, args.scale)
    net.load_state_dict(snapshot['net'])
    net.to(device)

    val(net, tqdm(val_loader), device, args)
