#coding: utf-8

from albumentations import (
    Compose, OneOf, ChannelDropout, ChannelShuffle, InvertImg, IAAAdditiveGaussianNoise, GaussNoise,
    Blur, MotionBlur, MedianBlur, GaussianBlur, CLAHE, IAASharpen, IAAEmboss, IAAAdditiveGaussianNoise,
    RandomBrightnessContrast, HueSaturationValue, RandomRain, RandomSunFlare, RandomFog,
    HueSaturationValue, GaussNoise, IAASharpen, IAAEmboss, OneOf, Compose, Resize, InvertImg, RandomRotate90,
    Flip, Transpose, ShiftScaleRotate, ShiftScaleRotate, OpticalDistortion, GridDistortion, IAAPiecewiseAffine
)

def pantograph_aug(p=1):
    aug = Compose([
        # RandomRotate90(),
        # 翻转
        # Flip(),
        # Transpose(),
        OneOf([
            # 高斯噪点
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            # 模糊相关操作
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     # 畸变相关操作
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        HueSaturationValue(p=0.3),
    ], p=1.0)

    return aug


augmentaion_factory = {
    "pantograph": pantograph_aug,
}