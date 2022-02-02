import torch
import torch.nn as nn


def conv5x5(in_planes: int, out_planes: int, stride: int = 1,  padding: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """5x5 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding,
                    groups = groups, bias = False, dilation = dilation)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """1x1 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1, groups: int = 1) -> nn.Conv2d:
    """2x2 deconv"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, padding=padding,
                              output_padding=0, groups=groups, bias=False)


def create_res_block(repeat_num: int = 1, in_planes: int = 1, stride: int = 1):
    layers = []
    for _ in range(repeat_num):
        layers.append(ResidualBlock(in_planes=in_planes, stride=stride))
    return nn.Sequential(*layers)


def fusion_module(planes):
    layers = []
    layers.append(Conv_BN_ReLU(in_planes=planes*2, out_planes=planes, kernel_size=1, padding=0))
    layers.append(Conv_BN_ReLU(in_planes=planes, out_planes=planes, kernel_size=3, padding=1))
    return nn.Sequential(*layers)


class Conv_BN_ReLU(nn.Module):
    """
        make 1x1 / 3x3 / 5x5 Conv + BN + ReLU sequential layer adaptively
    """
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 groups: int = 1, dilation: int = 1, norm_layer=None):
        super(Conv_BN_ReLU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if kernel_size == 1:
            self.conv = conv1x1(in_planes, out_planes, stride=stride, padding=padding)
        elif kernel_size == 3:
            self.conv = conv3x3(in_planes, out_planes, stride=stride, padding=padding, groups=groups, dilation=dilation)
        elif kernel_size == 5:
            self.conv = conv5x5(in_planes, out_planes, stride=stride, padding=padding, groups=groups, dilation=dilation)
        else:
            raise NotImplementedError("kernel_size only can be 1, 3, 5")
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Deconv_BN_ReLU(nn.Module):
    """
        make 2x2 Deconv + BN + ReLU sequential layer adaptively
    """
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 groups: int = 1, norm_layer=None):
        super(Deconv_BN_ReLU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if kernel_size == 2:
            self.deconv = deconv2x2(in_planes, out_planes, stride=stride, padding=padding, groups=groups)
        else:
            raise NotImplementedError("kernel_size only can be 2")
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
        ResNet基本模块
    """
    def __init__(self, in_planes: int, stride: int = 1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_planes, in_planes, stride)
        self.bn1 = norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes, in_planes, stride)
        self.bn2 = norm_layer(in_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    input_ = torch.randn((1, 3, 576, 1024))
    a = Conv_BN_ReLU(3, 32, kernel_size=5, padding=2, stride=2)
    b = Conv_BN_ReLU(32, 64, kernel_size=3, padding=1, stride=2)
    # largest_futuremap = model_out[0][0][0]
    # print(largest_futuremap.shape)