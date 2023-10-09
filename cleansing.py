import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils import  dir_exists,remove_files
def data_process(data_path, name, patch_size, stride,save_path ):
    
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, 'train', "images")
        gt_path = os.path.join(data_path, 'train', "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    elif name=='finetone':
        img_path=os.path.join(data_path,'image')
        gt_path=os.path.join(data_path,'mask')
        file_list = list(sorted(os.listdir(img_path)))      
    img_list = []
    gt_list = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "CHASEDB1":
            if len(file) == 13:
                img = Image.open(os.path.join(data_path, file))
                gt = Image.open(os.path.join(
                    data_path, file[0:9] + '_1stHO.png'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
                
        elif name == "DCA1":
            if len(file) <= 7:
            # the groundtruth file names "<>_gt.pgm">7
                # this dataset has 137 dataset we use :100 for training
                img = cv2.imread(os.path.join(data_path, file), 0)
                gt = cv2.imread(os.path.join(
                    data_path, file[:-4] + '_gt.pgm'), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            
            img = cv2.imread(os.path.join(img_path, file), 0)
            if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                tail = "PNG"
            else:
                tail = "png"
            gt = cv2.imread(os.path.join(
                gt_path, "angio"+file[:-4] + "ok."+tail), 0)
            gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
            img = cv2.resize(
                img, (512, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f"save_picture/{i}img.png", img)
            cv2.imwrite(f"save_picture/{i}gt.png", gt)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name=='finetone':
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file.split('.')[0]+'.png'))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

    img_list = normalization(img_list)
    
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    save_patch(img_patch, save_path, "img_patch")
    save_patch(gt_patch, save_path, "gt_patch")


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    if name=="STARE":
        pad = nn.ConstantPad2d((0, 2, 0, 1), 0)
    else:
        _, h, w = img_list[0].shape
        pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save_patch(imgs_list, path, type):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            # print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            # print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    print(mean,std)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    from config import get_config
    args=get_config()
    os.makedirs(args.path_tar,exist_ok=True)
    for name in ['DRIVE','CHASEDB1','STARE','CHUAC','DCA1']:
        data_path=os.path.join(args.path_src,name)
        save_path=os.path.join(args.path_tar,name)
        data_process(data_path, args.name,
                 args.patch_size, args.stride, save_dict=save_path)
        
    data_path=os.path.join(args.path_src,'finetone')
    save_path=os.path.join(args.path_tar,'finetone')
    data_process(data_path, 'finetone',
                 args.patch_size, args.stride, save_dict=save_path)