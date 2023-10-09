import os
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from utils import get_instance,dir_exists,get_metrics,AverageMeter
import math

class FineToner:
    def __init__(self, model, CFG=None, loss=None):
        self.CFG = CFG
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        # self.model = nn.DataParallel(model.cuda())
        self.model=model.cuda()
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'])
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True

    def train(self,train_loader):
        print("begin training process, will train the epoch {}".format(self.CFG.epochs))
        for epoch in range(1, self.CFG.epochs + 1):
            self._train_epoch(epoch,train_loader)
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch,train_loader):
        self.model.train()
        self._reset_metrics()
        batch_number=len(train_loader)
        for data_iter_step,  (img, gt) in  enumerate(train_loader):
            # adjust lr 
            self._adjust_learning_rate( data_iter_step / batch_number + epoch)
            
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
            
        print('TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f}  |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values()))
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
    
    def _adjust_learning_rate(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.CFG.warmup_epochs:
            lr = self.CFG.lr * epoch / self.CFG.warmup_epochs 
        else:
            lr = self.CFG.min_lr + (self.CFG.lr - self.CFG.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.CFG.warmup_epochs) / (self.CFG.epochs - self.CFG.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    
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