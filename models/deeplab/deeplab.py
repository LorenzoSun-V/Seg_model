import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from models.deeplab.aspp import build_aspp
from models.deeplab.decoder import build_decoder
from models.deeplab.backbone import build_backbone
from utils.loss import *


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 criterion=None, pretrained=True, norm_layer=None, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if norm_layer is None:
            BatchNorm = nn.BatchNorm2d
        else:
            BatchNorm = norm_layer

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.loss = criterion
        self.freeze_bn = freeze_bn

    def forward(self, input, gt=None):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if gt is None:
            return x
        else:
            loss = self.loss(x, gt)
            return x, loss

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.SyncBatchNorm) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.SyncBatchNorm) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


def deeplab(cfg, weight=None):
    cfg = edict(cfg)
    criterion = SegmentationLosses(weight=weight, cuda=torch.cuda.is_available()).build_loss(mode=cfg.loss)
    if cfg.bn == 'bn':
        norm_layer = None
    elif cfg.bn == 'syncbn':
        norm_layer = nn.SyncBatchNorm
    else:
        raise RuntimeError('Unknown bn type, check param model.bn in yaml. Only support bn and syncbn')
    model = DeepLab(backbone=cfg.backbone, num_classes=cfg.num_classes, criterion=criterion,
                    pretrained=cfg.pretrained, norm_layer=norm_layer)
    return model


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())