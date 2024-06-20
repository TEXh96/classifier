#coding: utf-8  resnet_1280_dc
from .nets.pp_lcnet import PPLCNet_x1_5
from .nets.mobilenetone import Mobilnetone18_1280x1280
from .nets.resnet_1280_dc import ResNet18_1280x1280
from .nets.resnet import ResNet18
from .nets.mobilenetv2 import MobileNetV2
from .nets.mobilenetv3 import MobileNetV3_my
from .nets.mobilenetv4 import MobileNetV4_my

net_factory = {
    'resnet18': ResNet18,
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3_my,
    'resnet18_1280': ResNet18_1280x1280,
    'moblibnetone_1280': Mobilnetone18_1280x1280,
    'Plcnet_56_448': PPLCNet_x1_5,
    'mobilenetv4':MobileNetV4_my,
}