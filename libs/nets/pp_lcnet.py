import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PPLCNet_x0_25", "PPLCNet_x0_35", "PPLCNet_x0_5", "PPLCNet_x0_75", "PPLCNet_x1_0",
    "PPLCNet_x1_5", "PPLCNet_x2_0", "PPLCNet_x2_5"
]

# NET_CONFIG = {
    # "blocks2":
    # #k, in_c, out_c, s, use_se
    # [[3, 16, 32, 1, False]],
    # "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    # "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    # "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                # [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                # [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    # "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
# }
NET_CONFIG_896 = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 2, False],[3, 32, 32, 2, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}



NET_CONFIG_448 = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 2, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}

NET_CONFIG_56_448 = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, (2,1), False]],
    "blocks3": [[3, 32, 64, (2,1), False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2,1), False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}




NET_CONFIG = NET_CONFIG_56_448

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Hardswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hardsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=True) / 6.

class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            num_filters,
        )
        self.hardswish = Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.mul(identity, x)
        return x


class PPLCNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 num_classes=1000,
                 dropout_prob=0.2,
                 class_expand=256):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = Hardswish()
        self.dropout = nn.Dropout(dropout_prob)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc = nn.Linear(self.class_expand, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        #112
        x = self.blocks2(x)
        x = self.blocks3(x)
        #56
        x = self.blocks4(x)
        #28
        x = self.blocks5(x)
        #14
        x = self.blocks6(x)
        #7

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def PPLCNet_x0_25(**kwargs):
    """
    PPLCNet_x0_25
    """
    model = PPLCNet(scale=0.25, **kwargs)

    return model


def PPLCNet_x0_35(**kwargs):
    """
    PPLCNet_x0_35
    """
    model = PPLCNet(scale=0.35, **kwargs)

    return model


def PPLCNet_x0_5(**kwargs):
    """
    PPLCNet_x0_5
    """
    model = PPLCNet(scale=0.5, **kwargs)

    return model


def PPLCNet_x0_75(**kwargs):
    """
    PPLCNet_x0_75
    """
    model = PPLCNet(scale=0.75, **kwargs)

    return model


def PPLCNet_x1_0(**kwargs):
    """
    PPLCNet_x1_0
    """
    model = PPLCNet(scale=1.0, **kwargs)

    return model


def PPLCNet_x1_5(**kwargs):
    """
    PPLCNet_x1_5
    """
    model = PPLCNet(scale=1.5, **kwargs)

    return model


def PPLCNet_x2_0(**kwargs):
    """
    PPLCNet_x2_0
    """
    model = PPLCNet(scale=2.0, **kwargs)

    return model


def PPLCNet_x2_5(**kwargs):
    """
    PPLCNet_x2_5
    """
    model = PPLCNet(scale=2.5, **kwargs)

    return model
    
    
# torchnet = PPLCNet_x2_5(num_classes=2,
                 # dropout_prob=0.2,
                 # class_expand=250)
# print("torchnet: ",torchnet)
