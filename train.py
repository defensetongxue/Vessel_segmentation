from ruamel.yaml import safe_load
from bunch import Bunch
import yaml
from torch.utils.data import DataLoader
import models
from utils import vessel_dataset
from processer import Trainer
from utils import losses,get_instance, seed_torch
import os

def train(CFG, path_tar,dataset, batch_size):
    seed_torch()
    # I have temporarily abandoned the val_loader 
    # which means all the data will be used in training
    
    #generare data 
    train_dataset=None
    if dataset=='all':
        all_dataset=['DRIVE', 'CHASEDB1' ,'CHUAC', 'DCA1', 'STARE']
        for name in all_dataset:
            data_path=os.path.join(path_tar,name)
            if train_dataset is None:
                train_dataset = vessel_dataset(data_path)
                print(f"load {len(train_dataset)} from {name}")
            else:
                new_dataset=vessel_dataset(data_path)
                train_dataset=train_dataset+new_dataset
                print(f"load {len(new_dataset)} from {name}")
    else:
        data_path=os.path.join(path_tar,dataset)
        train_dataset = vessel_dataset(data_path)
    # generate data loader
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16)
    print("generate dataloader finish there is {} of size {}".format(len(train_loader),batch_size))
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG)
    
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG
    )
    trainer.train(train_loader)


if __name__ == '__main__':
    from config import get_config
    args=get_config()
    import warnings
    warnings.filterwarnings("ignore")
    with open('./config/default.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    train(CFG, args.path_tar,args.dataset, args.batch_size)