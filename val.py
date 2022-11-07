import torch
from torchvision import ops, utils
from torchvision.transforms import Compose, ToPILImage, PILToTensor
from tqdm import tqdm

from lib.model.efficientdet import EfficientDet
from lib.utils import transforms
from lib.utils.box import Box
from lib.utils.data import collate_fn, getDataLoader
from lib.utils.utils import getArgs, getDevice


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
    args = getArgs()
    device = getDevice()

    args.batch_size = 1
    transf = Compose([transforms.EfficientToTensor(),
                      transforms.EfficientNormalize(args.mean, args.std),
                      transforms.EfficientResize(args.size),
                      transforms.EfficientPad(args.size)])
    val_loader = getDataLoader(args.valset['root'], args.valset['annFile'], transforms,
                               batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    snapshot = torch.load(args.weight, map_location=torch.device('cpu'))

    net = EfficientDet(len(args.categories), eval(args.anchors['scales']), eval(args.anchors['ratios']),
                       args.anchors['levels'], args.scale_feat, args.i_c, args.compound_coef, args.scale)
    net.load_state_dict(snapshot['net'])
    net.to(device)

    toPILimage = ToPILImage()
    toTensor = PILToTensor()

    net.eval()
    progress = tqdm(val_loader)
    for ind, (image, boxes, classes, orgi_image) in enumerate(progress):
        with torch.no_grad():
            cla, reg, anc = net(image.to(device))
            cla = torch.sigmoid(cla)

            progress.set_description(f'Validating image {ind}')

            boxes = Box.decode(reg, anc)

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
                                               [args.categories[c.item()] for c in categories])

            output = toPILimage(output.to(torch.float) / 255)
            output.save(f'output/{ind}.png')
