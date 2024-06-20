'''
Author: your name
Date: 2022-03-15 16:17:33
LastEditTime: 2022-03-15 17:06:45
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /prob/scripts/forward.py
'''
#coding: utf-8
import sys
#sys.path.insert(0,'/home/cbpm/duchao/classifier')

sys.path.append("/home/cbpm/duchao/wm/classifier")
import os
import argparse
import json
import numpy as np
import cv2
import torch
# from config import get_cfg 
from libs.net_factory import net_factory
# from ..utils.tools import *
from utils import *
from libs.dataset_factory import  preprocessing_factory
import shutil
import tqdm
import time

torch.cuda.set_device(1)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


class Classifier(object):
    
    def __init__(self, model_file, labelmap_file, use_gpu=True):
        Net = net_factory['mobilenetv2']
        self.label_map = parser_labelmap(labelmap_file)
        self.net = Net(num_classes=len(self.label_map))
        ckpt = torch.load(model_file)
        self.net.load_state_dict(ckpt['model'])
        self.preprocess = preprocessing_factory['yzd']
        
        if use_gpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.net = self.net.to(self.device)
        self.net.eval()
    
    def forward(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.preprocess(image=img)["image"]
        img_tensor = img_tensor.unsqueeze(0).type(torch.FloatTensor)
        img_tensor = img_tensor.to(self.device)
        # print(img_tensor.size())
        output = self.net(img_tensor)
        probs = torch.nn.functional.softmax(output)
        pred = output.argmax(dim=1, keepdim=True).item()
        # print("output: {}, probs: {}, pred: {}".format(output, probs, pred))
        
        return pred


if __name__ == '__main__':
        model_file = '/home/cbpm/duchao/wm/classifier/tasks/Tantou_yzd/checkpoints/mobilenetv2_final.pth'
        labelmap_file = '/home/cbpm/wangj/project/zhudaokeji/prob/scripts/label_map.json'
        classifier = Classifier(model_file, labelmap_file)
        img = cv2.imread('/home/cbpm/duchao/wm/classifier/微信截图_20220310165549.png')
        classifier.forward(img)