#coding: utf-8
# from .datasets.pantograph import PantographDataset
from .datasets.classification_dataset import ClassificationDataset,  ClassificationDataset_1280_1280,ClassificationDataset_448_448,ClassificationDataset_56_448
from .datasets.image_augmentation import *

dataset_factory = {
    # "pantograph": PantographDataset,
    # "sdg_blur": PantographDataset,
    "ZhoneChe_sdg_blur_check": ClassificationDataset,
    "ZhongChe_sdg_integrity": ClassificationDataset,
    "ShenYang_ZhuiYuanQu": ClassificationDataset,
    "ZhongChe_sdg_scene": ClassificationDataset,
    'ZhongChe_sdg_integrity_2class': ClassificationDataset,
    'eye_date_224_224': ClassificationDataset,
    'eye_date_224_224': ClassificationDataset,
    'classe_tt': ClassificationDataset,
    'SaveVersion':ClassificationDataset,
    'USD':ClassificationDataset,
    'class_crop':ClassificationDataset,
    'General':ClassificationDataset,
    'guohui':ClassificationDataset_448_448,
    'safeline':ClassificationDataset_56_448,
}

augumentation_factory = {
    "ZhoneChe_sdg_blur_check": augmentation_classfication(),
    "ZhongChe_sdg_integrity": augmentation_classfication(),
    "ZhongChe_sdg_scene": augmentation_classfication(),
    "ShenYang_ZhuiYuanQu":None,
    "ZhongChe_sdg_integrity_2class": augmentation_classfication(),
    "eye_date_224_224": augmentation_classfication(),
    'classe_tt': augmentation_classfication(),
    'SaveVersion':augmentation_classfication(),
    'USD':augmentation_classfication(),
    'class_crop':augmentation_classfication(),
    'General':augmentation_classfication(),
    'guohui':augmentation_classfication(),
    'safeline':augmentation_classfication(),
}

preprocessing_factory = {
    "ZhoneChe_sdg_blur_check": preprocessing(),
    "ZhongChe_sdg_integrity": preprocessing(),
    "ShenYang_ZhuiYuanQu": preprocessing(),
    "ZhongChe_sdg_scene": preprocessing(),
    "ZhongChe_sdg_integrity_2class": preprocessing(),
    "eye_date_224_224": preprocessing(),
    'classe_tt': preprocessing(),
    'SaveVersion':preprocessing(),
    'USD':preprocessing(),
    'class_crop':preprocessing(),
    'General':preprocessing(),
    'guohui':preprocessing(size=[896, 896]),
    'safeline':preprocessing(size=[448, 56]),#height  width
}

# print(dataset_factory)