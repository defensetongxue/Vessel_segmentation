## NOTE
2023/03/07: in the orignal dataset of DRIVE, the labels of test dataset in note avaible. I have download them in [GitHubLink](https://github.com/Libo-Xu/DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/tree/master/DRIVE/test/1st_manual)

2023/03/07: I temporarily abandon the val_loader which means all of the data in training folder will be used in the training process
## confusion


2023/03/08: In the cleansing for STARE, i use padding impled by myself, there may be somthing wrong. I have notice that when generating test dataset in `cleansing.py` we call the get_squre to generate squre image which i do not understand the reason

2032/03/08 In the utils/tools.py L82   f1 = 2 * pre * sen / (pre + sen) has an warning invalid value encountered in double_scars. which means it is possibly divdde zero , this will happened when exact path has no vessel, which is pretty common
## TODO
2023/03/07: fix the code style of the cleasing process in main.py
 
