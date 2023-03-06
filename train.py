import argparse
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses,get_instance, seed_torch


def main(CFG, data_path, batch_size, with_val=False):
    seed_torch()
    if with_val:
        train_dataset = vessel_dataset(data_path, mode="training", split=0.9)
        val_dataset = vessel_dataset(
            data_path, mode="training", split=0.9, is_val=True)
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader if with_val else None
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', "./dataset/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=64,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()

    with open('default.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    main(CFG, args.dataset_path, args.batch_size, args.val)