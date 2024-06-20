#coding: utf-8

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

#TODO: resize, mean, std值通过参数传进来更通用  A.Resize(width=256, height=256)
def preprocessing(size=[224, 224], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
     # 归一化：(value/255 - mean) / std
    transform = A.Compose([
        A.Resize(width=size[1], height=size[0]),
        A.Normalize(mean=mean, std=std),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #imagenet rgb image format
        ToTensorV2(),
    ])

    return transform



# 分类任务数据增强参考，不同任务应该做适当调整
def augmentation_classfication():
    augmentation = A.Compose([

        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=1),

        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.5),

        # A.OneOf([
        #     A.MotionBlur(p=.2),
        #     A.MedianBlur(blur_limit=3, p=0.1),
        #     A.Blur(blur_limit=3, p=0.1),
        # ], p=0.5),

        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.2),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),

        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(distort_limit=0.015, p=0.5),
            # A.IAAPiecewiseAffine(p=0.5),
        ], p=1),

        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10,p=0.5),
    ])

    return augmentation


if __name__ == "__main__":
    transform = preprocessing()
    augmentation = augmentation_classfication()
    aug = augmentation_classfication()
    # print(augmentation)
    A.save(augmentation, 'aug.json')
    augmentation = A.load('aug.json')

    aug_dict = A.to_dict(aug)
    print("aug_dict: ", aug_dict)
    while True:
        image = cv2.imread("temp/blur/20210718_06_08_1110_G901_001000_blur_1_0.581.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
        # print(image[0][0][0])
        # transform = preprocessing()
        # augmentation = augmentation_classfication()
        image = augmentation(image=image)["image"]
        cv2.imwrite("result.png", image)
        input()

        # image = transform(image=image)["image"]
        # print(image.shape)
        # print(image[0][0][0])