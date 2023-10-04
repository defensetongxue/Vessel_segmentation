import argparse
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_src', default="../autodl-tmp/vsdata_src", type=str,
                        help='the path of dataset')
    parser.add_argument('--path_tar', default="../autodl-tmp/vsdata_tar", type=str,
                        help='the path of dataset')
    parser.add_argument( '--dataset', default="all", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1'])
    parser.add_argument( '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('--stride', default=6,
                        help='the stride of image partition')
    parser.add_argument( '--batch_size', default=1024,
                        help='batch_size for trianing and validation')
    args = parser.parse_args()
    return args