'''
Author: Siyuan Li
Date: 2022-01-18 10:36:37
LastEditors: Siyuan Li
LastEditTime: 2022-05-07 14:24:11
FilePath: /classifier/train.py
Description: 
Email: 497291093@qq.com
Copyright (c) 2022 by Siyuan Li - ZHONGCHAO XINDA, All Rights Reserved. 
'''
#coding: utf-8

import os
import argparse
import albumentations as A
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from config import get_cfg 
from libs.dataset_factory import dataset_factory, augumentation_factory, preprocessing_factory
from libs.net_factory import net_factory
from utils.tools import *

torch.cuda.set_device(0)
print(torch.cuda.is_available())

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True



def train(model, device, train_loader, optimizer, criterion, epoch, log_iter=1000):
    model.train() #加上该句标明是训练阶段，BN和dropout保留。model.eval()作用类似
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader): #data, target比inputs/labels更通用
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("data: ",data.shape)
        output = model(data) #forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss
        if batch_idx % log_iter == log_iter-1:
            # print("batch_idx: ", batch_idx)
            print('Train Epoch: {} [{}/{}] loss: {}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
                                                                   running_loss/log_iter))
            running_loss = 0.0


def test(model, device, test_loader, criterion, epoch):
    global cfg
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # 保存当前测试集准确率最高的模型
    acc = 100.0 * correct / len(test_loader.dataset)
    if acc >= cfg['best_acc']:
        print("Saving current best model ...")
        print("acc: {}, best_acc: {}".format(acc, cfg['best_acc']))
        # print("model:\n", model)
        state = {
            "model": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if os.path.exists(cfg['current_model']):
            os.remove(cfg['current_model'])
        cfg['current_model'] = os.path.join(cfg['save_path'], '_{}_acc{:.2f}.pth'.format(cfg["train_date"], acc))
        torch.save(state, cfg['current_model'], _use_new_zipfile_serialization=False)
        torch.save(state, cfg['save_model'], _use_new_zipfile_serialization=False)
        cfg['best_acc'] = acc


def main(cfg):
    cfg["train_date"] = get_strdata()
    print(cfg['labelmap_file'])
    # try: #加载序列化的数据增强方法
        # aug = A.load(cfg['augmentation_json'])
        # preprocessing = A.load(cfg['preprocess_json'])
    # except: #加载配置文件中的数据增强方法
        # aug = augumentation_factory[cfg['task']]
        # preprocessing = preprocessing_factory[cfg['task']]
        # A.save(aug, cfg['augmentation_json'])
        # A.save(preprocessing, cfg['preprocessing_json'])
    aug = augumentation_factory[cfg['task']] #图像增强
    preprocessing = preprocessing_factory[cfg['task']] #图像预处理
    # A.save(aug, cfg['augmentation_json'])
    # A.save(preprocessing, cfg['preprocessing_json'])   
    print("aug: ", aug)
    print("preprocessing: ", preprocessing)

    DataSet=dataset_factory[cfg['task']]
    train_set = DataSet(cfg['annotations_train_txt'], cfg['labelmap_file'], augmentation=aug, preprocessing=preprocessing)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    test_set = DataSet(cfg['annotations_val_txt'], cfg['labelmap_file'], preprocessing=preprocessing)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])

    Net = net_factory[cfg['backbone']]
    label_map = parser_labelmap(cfg['labelmap_file'])
    print("len(label_map): ",len(label_map))
    net = Net(num_classes=len(label_map))
    print(net)
    
    #暂时不加载模型
    # if cfg['resume']:
        # if os.path.exists(cfg['resume_model']):
            # print("loading resume network: {}".format(cfg['resume_model']))
            # ckpt = torch.load(cfg['resume_model'])
            # net.load_state_dict(ckpt['model'])
            # cfg['best_acc'] = ckpt['acc']
    
    make_if_not_exit(cfg['save_path'])
    
    if cfg['use_gpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate']) 
    criterion = torch.nn.CrossEntropyLoss()  
    scheduler = StepLR(optimizer, 50, gamma=0.2)

    for epoch in range(0, cfg['num_epoch']):
        train(net, device, train_loader, optimizer, criterion, epoch, log_iter=cfg['log_iter'])
        test(net, device, test_loader, criterion, epoch)
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training...')
    parser.add_argument('-task', default=None, help='image classification task name')
    # parser.add_argument('-net', default=None, help='classification net')
    opt = parser.parse_args()
    # opt.task = 'ZhoneChe_sdg_blur_check' #分类任务名，一个项目任务对应配置文件中一个字典
    # opt.task = 'ShenYang_ZhuiYuanQu'
    # opt.task = 'ZhongChe_sdg_scene'
    # opt.task = 'eye_date_224_224'
    # opt.task = 'ZhongChe_sdg_integrity'
    # opt.task = 'ZhongChe_sdg_integrity_2class'
    # opt.task = 'classe_tt' 
    #opt.task= 'SaveVersion'
    # opt.task= 'CNY'
    # opt.task= 'guohui'
    opt.task= 'pibu'
    cfg = get_cfg(opt.task)
    print(cfg)

    main(cfg)