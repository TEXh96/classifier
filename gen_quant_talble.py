#coding: utf-8

'''
    生成、制作量化校正表及rknn test image
'''
import os
import argparse
import shutil
import cv2
from config import get_cfg 
from utils.tools import *


def main(cfg):
    with open(cfg['annotations_train_txt'], 'r') as f:
        lines = f.readlines()
        if len(lines) > 10:
            lines = lines[0:10]

        # copy images 
        quant_table_path = os.path.dirname(cfg['dataset_txt'])
        make_if_not_exit(quant_table_path)
        basenames = []
        for line in lines:
            file, _ = line.rstrip().split(' ')
            shutil.copy(file, quant_table_path)
            basenames.append(os.path.basename(file))
        
        # gen dataset.txt   
        with open(cfg['dataset_txt'], 'w+') as fwriter:
            for basename in basenames:
                fwriter.write(basename+ '\n')

        # save rknn test image
        file, _ = lines[0].rstrip().split(' ')
        image = cv2.imread(file)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(cfg['rknn_test_img'], image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training...')
    parser.add_argument('-task', default=None, help='image classification task name')
    # parser.add_argument('-net', default=None, help='classification net')
    opt = parser.parse_args()
    opt.task = 'ZhoneChe_sdg_blur_check' #分类任务名，一个项目任务对应配置文件中一个字典
    cfg = get_cfg(opt.task)

    main(cfg)
