#coding:utf-8

'''
    模型定义
'''

import torch.nn as nn
import torchvision.models as models
import math
import torch.utils.model_zoo as model_zoo

def ResNet18(num_classes=2, pretrained=True):
    resnet18 = models.resnet18(pretrained=pretrained)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    # resnet18.add_module("softmax", nn.Softmax2d)

    return resnet18


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_, self).__init__()
        
        #1280×1280
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#
        
        #640×640
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        #320×320
        self.layer2 = self._make_layer(block, 64, layers[0], stride=2)
        
        #160*160
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3_1 = self._make_layer(block, 128, layers[1])
        #80*80
        self.layer4 = self._make_layer(block, 128, layers[1], stride=2)
        #40*40
        self.layer5 = self._make_layer(block, 128, layers[1], stride=2)
   
   
        #20*20
        self.layer6 = self._make_layer(block, 256, layers[3], stride=2) 
        #10*10
        self.layer7 = self._make_layer(block, 256, layers[3], stride=1)
        # self.layer8 = self._make_layer(block, 256, layers[3])
        #5*5
        
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )   

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input: ",x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        # print("conv1: ",x.shape)
        x = self.layer1(x)
        # print("layer1: ",x.shape)
        x = self.layer2(x)
        # print("layer2: ",x.shape)
        x = self.layer3(x)
        
        # print("layer3: ",x.shape)
        x = self.layer4(x)
        # print("layer4: ",x.shape)
        x = self.layer5(x)
        # print("layer5: ",x.shape)
        x = self.layer6(x)
        # print("layer6: ",x.shape)
        x = self.layer7(x)
        
        # print("layer7: ",x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_(BasicBlock, [1, 2, 2, 2], **kwargs)
  #  if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ResNet18(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.resnet18 = resnet18_(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(256, 256)
       # self.resnet18.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)  ##hgx
        self.resnet18.avgpool =nn.AvgPool2d(kernel_size=5, stride=2, padding=0)
        self.net_add = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes)  ##hgx
            # nn.Softmax()
        )


    def forward(self, X):
        features = self.resnet18(X)
        output = self.net_add(features)

        return output


class ResNet18_1280x1280(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18_1280x1280, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.resnet18 = resnet18_(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(256, 256)
       # self.resnet18.conv1=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)  ##hgx
        self.resnet18.avgpool =nn.AvgPool2d(kernel_size=5, stride=2, padding=0)
        
        
        self.net_add = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes)  ##hgx
            # nn.Softmax()
        )


    def forward(self, X):
        features = self.resnet18(X)
        output = self.net_add(features)

        return output


class ResNet18_old(nn.Module):
    """基于ResNet18修改并构建类别为num_classes的分类网络"""

    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18_old, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 512)
        self.net_add = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        features = self.resnet18(X)
        output = self.net_add(features)

        return output



if __name__ == '__main__':
    model = ResNet18(num_classes=2)