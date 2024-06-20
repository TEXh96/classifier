#coding: utf-8
import os
import time
import json
import cv2
import random

def make_if_not_exit(dir):
    '''
    创建文件目录[若目录不存在]
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_file_list(root_path, postfix=None):
    '''
    获取root_path目录下的所有后缀名为postfix的文件
    '''
    file_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            filename = os.path.join(root, file)
            file_list.append(filename)

    if postfix:
        file_list = list(filter(lambda filename: filename.endswith(postfix), file_list))

    return file_list


def get_strdata():
    localtime = time.localtime(time.time())

    return time.strftime("%Y%m%d", localtime)

def parser_labelmap(json_file):
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print("error: {}".format(e))
        exit(-1)


if __name__ == '__main__':
    img_files = get_file_list('/mnt/10_AlgorithmData/ZNJT/ZhoneChe_sdg_blur_check/CleanData', '.jpg')
    save_path = '/data2/wangj/research/image_classification/classifier/tasks/ZhoneChe_sdg_blur_check/quant_table'
    make_if_not_exit(save_path)
    
    random.shuffle(img_files)
    if len(img_files) > 100:
        img_files = img_files[0:100]
    for img_file in img_files:
        img = cv2.imread(img_file)
        basename = os.path.basename(img_file)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(os.path.join(save_path, basename), img)
