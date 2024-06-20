#coding: utf-8
'''
    分类任务配置文件

    关于模型文件的约定：
        1.所有模型都存储在tasks/${task_name}/checkpoints/目录下
        2.模型文件命名：
            ${backbone}_final.pth #当前最优模型，加样训练及模型转换时都用这个模型
            ${backbone}_YYYYMMDD_accxxx.pth #特定日期下准确率为xxx的训练模型
            模型转换时存储的模型名为：${backbone}_final.xxx
'''

import os


cfg_ShenYang_person_fall_action = {
    # #dataset，该部分无需设置
    # 'annotations_train_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/train.txt',
    # 'annotations_test_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/val.txt',
    # 'labelmap_file': 'tasks/ZhoneChe_sdg_blur_check/dataset/label_map.json',

    # backbone
    'backbone': 'resnet18', 
    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}





cfg_ZhoneChe_sdg_blur_check = {
    # #dataset，该部分无需设置
    # 'annotations_train_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/train.txt',
    # 'annotations_test_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/val.txt',
    # 'labelmap_file': 'tasks/ZhoneChe_sdg_blur_check/dataset/label_map.json',

    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}




cfg_eye_date_224_224 = {
    # #dataset，该部分无需设置
    # 'annotations_train_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/train.txt',
    # 'annotations_test_txt': 'tasks/ZhoneChe_sdg_blur_check/dataset/val.txt',
    # 'labelmap_file': 'tasks/ZhoneChe_sdg_blur_check/dataset/label_map.json',

    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_ZhongChe_sdg_scene = {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}


cfg_ZhongChe_sdg_integrity = {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8,  #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_classe_tt =  {
    # backbone
    'backbone': 'mobilenetv3', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_SaveVersion =  {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_USD =  {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}



cfg_AUD=  {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}
cfg_class_crop=  {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}
cfg_General=  {
    # backbone
    'backbone': 'mobilenetv2', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_guohui=  {
    # backbone
    #'backbone': 'resnet18', 
    'backbone': 'Plcnet_448',
    'batch_size': 8,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-3,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}
cfg_safeline=  {
    'backbone': 'Plcnet_56_448',
    'batch_size': 8,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-3,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_pibu=  {
    # backbone
    'backbone': 'mobilenetv4', 

    'batch_size': 32,
    'num_workers': 8, #4
    'resume': True,
    'use_gpu': True,
    'learning_rate': 1e-5,
    'num_epoch': 1000,
    'log_iter': 20,

    'best_acc': 0.0,
}

cfg_factory = { # task: cfg
    # 'pantograph': cfg_pantograph, #受电弓完整性分类
    # 'sdg_blur': cfg_sdg_blur, #受电弓区域模糊判定
    'ZhoneChe_sdg_blur_check': cfg_ZhoneChe_sdg_blur_check,
    'ZhongChe_sdg_integrity': cfg_ZhongChe_sdg_integrity,
    'ShenYang_ZhuiYuanQu': cfg_ShenYang_person_fall_action,
    'ZhongChe_sdg_scene': cfg_ZhongChe_sdg_scene,
    'ZhongChe_sdg_integrity_2class': cfg_ZhongChe_sdg_integrity,
    'eye_date_224_224': cfg_eye_date_224_224,
    'classe_tt': cfg_classe_tt,
    'SaveVersion':cfg_SaveVersion,
    'class_AUD':cfg_AUD,
    'class_crop':cfg_class_crop,   
    'General':cfg_General, 
    'guohui':cfg_guohui,
    'safeline':cfg_safeline,
    'pibu':cfg_pibu,
}

def get_cfg(task_name):
    assert task_name in cfg_factory.keys(), "there is no {} task in...\n {}".format(task_name, cfg_factory)
    cfg = cfg_factory[task_name]

    annotations_train_txt = f'tasks/{task_name}/dataset/train.txt'
    annotations_val_txt = f'tasks/{task_name}/dataset/val.txt'
    labelmap_file = f'tasks/{task_name}/dataset/label_map.json'

    save_path = f'tasks/{task_name}/checkpoints'
    resume_model = os.path.join(save_path, f'{cfg["backbone"]}_final.pth')
    save_model = resume_model

    #模型移植相关
    onnx_model = save_model.replace('.pth', '.onnx')
    rknn_model = save_model.replace('.pth', '.rknn')
    dataset_txt = f'tasks/{task_name}/quant_table/dataset.txt'
    rknn_test_img = f'tasks/{task_name}/test.jpg'

    #数据增强序列化相关,存在json文件就用json文件中的数据增强方式
    augmentation_json = f'tasks/{task_name}/augmentation.json'
    preprocessing_json = f'tasks/{task_name}/preprocessing.json'

    cfg.update({
        'task': task_name,
        'annotations_train_txt': annotations_train_txt,
        'annotations_val_txt': annotations_val_txt,
        'labelmap_file': labelmap_file,
        'save_path': save_path,
        'resume_model': resume_model,
        'save_model': save_model, #模型测试及转换需要用到的模型
        'current_model': '', #当前训练阶段最优模型，save_model的副本
        'onnx_model': onnx_model,
        'rknn_model': rknn_model,
        'dataset_txt': dataset_txt,
        'rknn_test_img': rknn_test_img,
        'augmentation_json': augmentation_json, 
        'preprocessing_json': preprocessing_json,
    })

    return cfg


