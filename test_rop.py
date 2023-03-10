import os
from PIL import Image
from torchvision.transforms import Grayscale, Normalize, ToTensor
import torch
import numpy as np
import pickle
import torchvision.transforms.functional as TF
from utils import dir_exists,remove_files,ROP_dataset,get_instance
from torch.utils.data import DataLoader
import cv2
from cleansing import normalization,save_each_image
import models
from ruamel.yaml import safe_load
from bunch import Bunch
# cleansing the dataset for ROP
def cleansing(data_path):
    save_path=data_path+"_pro"
    dir_exists(save_path)
    remove_files(save_path)
    file_list=os.listdir(data_path)
    img_list=[] 
    assert len(file_list)<=100, "too many image in testing which may cause the stack overflow"
    for file in file_list:
        img=Image.open(os.path.join(data_path, file))
        img = Grayscale(1)(img)
        img_list.append(ToTensor()(img))
    img_list=normalization(img_list)
    save_each_image(img_list,path=f'{data_path}_pro',type="img", name='ROP')
    print(img_list[0].shape)

def testing(model,data_path,weight_path):
    data_path = data_path
    dir_exists('./save_picture_ROP')
    # remove_files('./save_picture_ROP')
    model=model.cuda()
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    test_dataset = ROP_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    for i,(img,_ )in enumerate(test_loader):
        H,W=512,512
        pre=model(img.cuda())
        img = TF.crop(img, 0, 0, H, W)
        pre = TF.crop(pre, 0, 0, H, W)

        img = img[0,0,...]
        pre = pre[0,0,...]

        predict = torch.sigmoid(pre).cpu().detach().numpy()
        cv2.imwrite(
            f"save_picture_ROP/{i}_img.png", np.uint8(img.cpu().numpy()*255))
        cv2.imwrite(
            f"save_picture_ROP/{i}_pre.png", np.uint8(predict*255))
        # cv2.imwrite(
        #     f"save_picture_ROP/pre_b{i}.png", np.uint8(predict_b*255))

# cleansing('./ROP')

with open("./config/default.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
weight_path="./saved/FR_UNet/checkpoint-epoch40.pth"
model = get_instance(models, 'model', CFG)
testing(model=model,
        data_path='./ROP',
        weight_path=weight_path)