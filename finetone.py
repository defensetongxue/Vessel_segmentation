import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from utils import vessel_dataset
from processer import FineToner
from utils import losses,get_instance, seed_torch
import os

def train(CFG, path_tar, batch_size):
    seed_torch()
    # I have temporarily abandoned the val_loader 
    # which means all the data will be used in training
    
    #generare data 
    train_dataset=None
    
    data_path=os.path.join(path_tar,'finetone')
    train_dataset = vessel_dataset(data_path)
    # generate data loader
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16)
    print("generate dataloader finish there is {} of size {}".format(len(train_loader),batch_size))
    model = get_instance(models, 'model', CFG)
    checkpoint = torch.load(
            os.path.join('pretrained/vessel_seg.pth'))
    loaded_state_dict = checkpoint['state_dict']
    model.load_state_dict(loaded_state_dict)
    loss = get_instance(losses, 'loss', CFG)
    
    trainer = FineToner(
        model=model,
        loss=loss,
        CFG=CFG
    )
    trainer.train(train_loader)


if __name__ == '__main__':
    from config import get_config
    args=get_config()
    with open('./config/finetone.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    train(CFG, args.path_tar,args.batch_size)