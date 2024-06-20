#coding: utf-8
import torch.nn as nn
import torchvision.models as models

# def MobileNetV2(pretrained=True, **kwargs):
#     model = models.mobilenet_v2(pretrained=True, **kwargs)

#     return model

def MobileNetV2(pretrained=True, num_classes=1000):
    model = models.mobilenet_v2(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True),
    )

    return model

# model = MobileNetV2(pretrained=True, num_classes=2)
