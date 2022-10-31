import torch
from tqdm import tqdm


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
