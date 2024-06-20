#coding: utf-8
import os
import argparse
import onnx
import torch
import torch.onnx
# from test import load_model
from config import get_cfg 
from libs.net_factory import net_factory
from libs.dataset_factory import dataset_factory, augumentation_factory, preprocessing_factory
from utils.tools import *

def load_model(cfg):
    Net = net_factory[cfg['backbone']]
    label_map = parser_labelmap(cfg['labelmap_file'])
    net = Net(num_classes=len(label_map))
   
    # cfg['save_model']= "tasks/CNY/checkpoints/mobilenetv2_final.pth"
    print("cfg['save_model']: ",cfg['save_model'])
    ckpt = torch.load(cfg['save_model'])
    print("cfg['save_model']: ",cfg['save_model'])
    import pdb
    pdb.set_trace()
    net.load_state_dict(ckpt['model'])

    return net


def convert_to_onnx(net, output_name):
    input_names = ['data']
    input = torch.randn(1, 3, 224, 224) #TODO:size需要通过配置文件传进来
    output_names = ['output']
    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names,opset_version = 12)


def main(cfg):
    net = load_model(cfg)
    net.eval()

    print("----------",cfg['onnx_model'])
    convert_to_onnx(net, cfg['onnx_model'])
    print("convert_to_onnx finished !!!")

    onnx_model = onnx.load(cfg['onnx_model'])
    onnx.checker.check_model(onnx_model)
    print("checked !!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default=None, help='image classification task name')
    opt = parser.parse_args()
    # opt.task = 'ZhoneChe_sdg_blur_check'
    # opt.task = 'ZhongChe_sdg_integrity'
    # opt.task = 'eye_date_224_224'
    # opt.task = 'SaveVersion'
    # opt.task = 'class_crop'
    opt.task= 'General'
    # opt.task = 'ZhongChe_sdg_integrity_2class'
    cfg = get_cfg(opt.task)
# 
    main(cfg)