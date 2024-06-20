#coding: utf-8

import os
import argparse
import json
import numpy as np
import cv2
import torch
from config import get_cfg 
from libs.net_factory import net_factory
from utils.tools import *
from libs.dataset_factory import  preprocessing_factory
import shutil
import tqdm
import time

torch.cuda.set_device(0)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def load_model(cfg):
    Net = net_factory[cfg['backbone']]
    label_map = parser_labelmap(cfg['labelmap_file'])
    net = Net(num_classes=len(label_map))
    print(label_map)
    print(cfg['save_model'])
    ckpt = torch.load(cfg['save_model'])
    net.load_state_dict(ckpt['model'])

    return net

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def main(opt):
    net = load_model(cfg)
    if cfg['use_gpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    net = net.to(device)
    net.eval()

    label_map = parser_labelmap(cfg['labelmap_file'])
    print("label_map: ", label_map)
    preprocess = preprocessing_factory[cfg['task']]

    idx2label = {value: key for key, value in label_map.items()}
    print(idx2label)

    # # 测试单张
    # img_file = cfg["rknn_test_img"]
    # img_file = "/home/cbpm/duchap/project_temp/image_classification/classifier/tasks/CleanData/20211230/simple/img0_2418_0.684264.jpg"
    # img = cv2.imread(img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_tensor = preprocess(image=img)["image"]
    # img_tensor = img_tensor.unsqueeze(0).type(torch.FloatTensor)
    # img_tensor = img_tensor.to(device)

    # input_data = img_tensor.cpu().numpy().reshape(3*224*224)
    # # with open("input_data.txt", "w") as f:
        # # for i in input_data:
            # # f.write("%f\n"%i)

    # output = net(img_tensor)
    # probs = torch.nn.functional.softmax(output)
    # pred = output.argmax(dim=1, keepdim=True).item()
    # print("output: {}, probs: {}, pred: {}".format(output, probs, pred))





    # 批量测试
    file_path = r'/data2/duchao/classifier_20221118/images/hebing/B'
    # file_path = r"W:\Data\ZNJT\ZhongChe_sdg_integrity\CleanData\20211102\missing"
    img_files = get_file_list(file_path, ".bmp")
    #img_files = get_file_list(file_path)
    # img_files = ['/data2/wangj/temp/integrity1_input_1/img0_1109.jpg']
    # img_files = img_files[0:500]
    # debug_dir = 'temp/ShenYang_person_fall_action'
    #debug_dir = '/home/cbpm/duchao/classifier/train_datasets/SaveVersion/test'
    debug_dir = file_path + '_result'
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    make_if_not_exit(debug_dir)
    total, correct = 0, 0
    for idx in tqdm.tqdm(range(len(img_files))):
    #while(True):
        img_file = img_files[idx ]
        label = os.path.basename(os.path.dirname(img_file))

        if label not in label_map.keys():
            label=''
        else:
            label=label_map[label]
        # print(img_file, label, label_map[label])
        img = cv2.imread(img_file)
        # cv2.imwrite(os.path.join(debug_dir, "temp.jpg"), img)
        if img is None:
            continue
        total += 1
        # start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(image=img)["image"]
        img_tensor = img_tensor.unsqueeze(0).type(torch.FloatTensor)
        img_tensor = img_tensor.to(device)
        output = net(img_tensor)
        print("output: ",output)
        np_output = output.detach().cpu().numpy()
        
        # cls_output = torch.from_numpy(np_output[0][:2]).to(device)
        # mh_output = torch.from_numpy(np_output[0][3:]).to(device)
        
        cls_output = output[:,:2]
        mh_output = output[:,3:]
        # print("cls_output: ",cls_output)
        # print("mh_output: ",mh_output)
        # import pdb
        # pdb.set_trace()
        
        
        
        # print("time cost: ", time.time()-start)
        probs = torch.nn.functional.softmax(cls_output)
        mh_probs = torch.nn.functional.softmax(mh_output)
        print("cls score: ",probs)
        print("mh score: ",mh_probs)
        
        pred = cls_output.argmax(dim=1, keepdim=True).item()
        print("file:{}, probs:{}, pred:{} \n".format(img_file, probs, pred))
        if str(pred) == label:
            correct += 1
                
        else:
            print("file:{}, probs:{}, pred:{} \n".format(img_file, probs, pred))
            # import pdb
            # pdb.set_trace()
            # save_path = os.path.join(debug_dir, idx2label[str(pred)])
            # make_if_not_exit(save_path)
            # score = probs.cpu().detach().numpy().squeeze()[pred]
            # basename = os.path.basename(img_file)
            # shutil.copyfile(img_file, os.path.join(save_path, basename))
    
        print("acc: {}/{}, {:.3f}".format(correct, total, correct*1.0/total))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification test...')
    parser.add_argument('-task', default=None, help='image classification task name')
    opt = parser.parse_args()
    # opt.task = 'ZhongChe_sdg_integrity'
    # opt.task = 'ZhoneChe_sdg_blur_check'
    # opt.task = 'ShenYang_person_fall_action'
    # opt.task = 'ZhongChe_sdg_scene'
    opt.task = 'safeline'
    # opt.task= 'classe_tt'
    cfg = get_cfg(opt.task)
    print("cfg: ",cfg)

main(cfg)