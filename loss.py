import torch
def dice_loss(pred, target):
    smooth = 1.
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    dice = dice.mean()
    return 1 - dice
