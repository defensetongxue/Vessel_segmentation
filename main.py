import argparse
from bunch import Bunch
from ruamel.yaml import safe_load
from cleansing import data_process
from train import train
import os 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="all", type=str,
                        help='dataset used, if dataset==all, we will used all of the dataset')
    parser.add_argument( '--batch_size', default=64,
                        help='batch_size for trianing and validation')
    parser.add_argument( '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('--stride', default=6,
                        help='the stride of image partition')
    parser.add_argument( '--cleansing', default=False,
                        help='if do the cleansing task, note that each data should to first')
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")    
    # there is totally 5 data DRIVE CHASEDB1 CHUAC DCA1 STARE
    if args.cleansing:
        if args.dataset=='all':
            all_dataset=['DRIVE', 'CHASEDB1' ,'CHUAC', 'DCA1', 'STARE']
            for name in all_dataset:
                print("process and generate patches for dataset {}".format(name))
                data_path=os.path.join('../autodl-tmp/datasets_vessel',name)
                data_process(data_path,name,args.patch_size,args.stride,"training")
                data_process(data_path,name,args.patch_size,args.stride,"test")
                print("finish cleansing for {}".format(name))
        else:
            print("process and generate patches for dataset {}".format(args.dataset))
            data_path=os.path.join('../autodl-tmp/datasets_vessel',args.dataset)
            data_process(data_path,args.dataset,args.patch_size,args.stride,"training")
            data_process(data_path,args.dataset,args.patch_size,args.stride,"test")

            print("finish cleansing for {}".format(args.dataset))
    with open('./config/default.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
    train(CFG, args.dataset, args.batch_size)