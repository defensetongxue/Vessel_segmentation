import argparse
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from utils import vessel_dataset
from processer import Trainer
from utils import losses,get_instance, seed_torch
import os

def train(CFG, dataset, batch_size):
    seed_torch()
    # I have temporarily abandoned the val_loader 
    # which means all the data will be used in training
    
    #generare data 
    train_dataset=None
    if dataset=='all':
        all_dataset=['DRIVE', 'CHASEDB1' ,'CHUAC', 'DCA1', 'STARE']
        for name in all_dataset:
            data_path=os.path.join('../autodl-tmp/datasets_vessel',name)
            if train_dataset is None:
                train_dataset = vessel_dataset(data_path, mode="training")
            else:
                train_dataset=train_dataset+vessel_dataset(data_path, mode="training")
    else:
        data_path=os.path.join('../autodl-tmp/datasets_vessel',dataset)
        train_dataset = vessel_dataset(data_path, mode="training")
    # generate data loader
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DRIVE", type=str,
                        help='dataset used, if dataset==all, we will used all of the dataset')
    parser.add_argument( '--batch_size', default=64,
                        help='batch_size for trianing and validation')
    args = parser.parse_args()
    # there is totally 5 data DRIVE CHASEDB1 CHUAC DCA1 STAGE
    
    with open('./config/default.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    train(CFG, args.dataset, args.batch_size)