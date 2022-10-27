import torch


def val(net, dataloader, criterion, device):
    net.eval()
    epoch_loss = []
    for image, boxes, classes in dataloader:
        with torch.no_grad():
            output = net(image.to(device))
            loss = sum(criterion(*output, boxes, classes))

            epoch_loss.append(loss.item())
    return sum(epoch_loss)
