#coding: utf-8
import os
import cv2
from utils import *

rects = {
    "shanghai": [719, 613, 1031, 705],
    "nanjing": [459, 618, 685, 700],
}


def get_img_info(file):
    infos = file.split('_')
    loc = infos[0].split('-')[-1]
    
    return loc, infos[2]

    
if __name__ == '__main__':
    src_path = r'/media/cbpm2016/D/wangj/temp/prob/images'
    save_path = src_path + '_crop'
    
    img_files = get_file_list(src_path, 'png')
    # img_files = [img_file for img_file in img_files if 'shanghai' in img_file]
    img_files.sort()
    
    for img_file in img_files:
        img = cv2.imread(img_file)
        loc, camID = get_img_info(img_file)
        rect = rects[loc]
        img_roi = img[rect[1]:rect[3], rect[0]:rect[2], :]
        path = os.path.join(save_path, loc, camID)
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, os.path.basename(img_file)), img_roi)
    
    # img = cv2.imread('20220307-shanghai_6rza_1_20220301_092342_001000.png')
    # print(img.shape)
    # img_roi = img[613:705, 719:1031, :]
    # print(img_roi.shape)
    # cv2.imwrite("test.png", img_roi)
    