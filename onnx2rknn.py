# coding: utf-8
# import sys
# sys.path.insert(0, "/data1/anaconda3/envs/wangj_rknn/bin/python")
import numpy as np
import cv2
import argparse
from rknn.api import RKNN
from config import get_cfg 

def main(cfg):
    onnx_file = cfg['onnx_model']
    rknn_file = cfg['rknn_model']
    print("onnx_file: ", onnx_file)
    print("rknn_file: ", rknn_file)
    

    rknn = RKNN()
    print('--> config model')
    #rknn.config优先调整通道，再做归一化
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]], reorder_channel='2 1 0')
    #rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]], reorder_channel='0 1 2')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_file)
    if ret != 0:
        print('Load resnet50v2 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=cfg['dataset_txt'])
    if ret != 0:
        print('Build rknn model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_file)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(cfg['rknn_test_img'])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    
    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[0]
    print("x: ", x)
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    print("outputs: ",outputs)
    print('done')
    rknn.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification training...')
    parser.add_argument('-task', default=None, help='image classification task name')
    opt = parser.parse_args()
    # opt.task = 'ZhoneChe_sdg_blur_check'
    # opt.task =  'ZhongChe_sdg_integrity'
    opt.task = 'ZhongChe_sdg_scene'
    # opt.task = 'ZhongChe_sdg_integrity_2class'
    cfg = get_cfg(opt.task)

    main(cfg)



