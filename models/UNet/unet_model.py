""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from easydict import EasyDict as edict
from utils.loss import *
from utils.model_utils import load_weights


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, criterion=None, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss = criterion
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, gt=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if gt is not None:
            loss = self.loss(logits, gt)
            return logits, loss
        else:
            return logits


def unet(cfg, weight=None):
    cfg = edict(cfg)
    criterion = SegmentationLosses(weight=weight, cuda=torch.cuda.is_available()).build_loss(mode=cfg.loss)
    model = UNet(n_channels=cfg.n_channels, n_classes=cfg.num_classes, criterion=criterion, bilinear=True)
    if cfg.pretrained:
        load_weights(model, cfg.pretrained_model_url)
    return model


