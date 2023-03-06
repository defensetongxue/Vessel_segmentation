from .dataloder import generate_dataloader
from .models import Unet as build_model
import torch.optim as optim
import torch

train_dataloader,val_dataloader=generate_dataloader()
model=build_model()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)