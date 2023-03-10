# blood vessel segmentation module
This is a subproject for ROP dignoise, reimplement FR-UNet for blood vessel segmentation task. Most code are from the official repository of the [FR-UNet](https://github.com/lseventeen/FR-UNet), thanks a lot.

The main contribution of the repository is:
1. Compose 5 datasets( DRIVE,CHASEDB1,STARE,CHUAC, and DCA1 ) in the training process which can improve the robustness of the model to fit the other dataset.
2. Provide a interface for the other project by moving the `./VesselSegModule` to the main project path, After you get the pretrained model by the other file and copy it to `VesselSegModule/checkpoint` and rename it with  `best.pth`

the usage of the main file:

`main.py` This file provides training process for  5 datasets( DRIVE,CHASEDB1,STARE,CHUAC, and DCA1 )

`train.py` This file is the orignal traning process for the single dataset

`test.py` This file is the orignal test process for the single dataset, you can evaluate your model here.

Thank for the FR-UNet again. If you have any question, you can give an issue and I am glad to help you.

## NOTE
2023/03/07: in the orignal dataset of DRIVE, the labels of test dataset in note avaible. I have download them in [GitHubLink](https://github.com/Libo-Xu/DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/tree/master/DRIVE/test/1st_manual)

2023/03/07: I temporarily abandon the val_loader which means all of the data in training folder will be used in the training process

