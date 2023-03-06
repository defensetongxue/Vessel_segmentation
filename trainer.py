import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from .utils import get_instance,dir_exists,get_metrics,AverageMeter


class Trainer:
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'], start_time)
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            self._train_epoch(epoch)
            if self.val_loader is not None :
                results = self._valid_epoch(epoch)
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()
        for img, gt in self.train_loader:
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=True):
                pre = self.model(img)
                loss = self.loss(pre, gt)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
           
            self.total_loss.update(loss.item())
            
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
            print(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f}  |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values()))
            
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
                
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }