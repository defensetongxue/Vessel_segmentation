'''
the file is the utils funtion for training process
'''
import torch
from .loss import dice_loss as loss_function

def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(dataloader.dataset)
    return train_loss

# Define validation function
def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * images.size(0)
        val_loss /= len(dataloader.dataset)
    return val_loss

