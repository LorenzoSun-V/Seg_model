import torch
import torch.nn as nn
from models.modules.common import *
from utils.loss import *
from easydict import EasyDict as edict
from models.load_weights import load_weights


class HaoSegNet(nn.Module):
    def __init__(self, num_classes=2, criterion=None,
                 channel_list=[32, 64, 128, 256, 256, 256],
                 res_block_list=[2, 2, 2, 2, 2],
                 norm_layer=None):
        super(HaoSegNet, self).__init__()
        assert num_classes > 0, 'num_classes must be greater than 0'
        if num_classes <= 2:
            feature_map_out = 1
        else:
            feature_map_out = num_classes
        self.loss = criterion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # self._norm_layer = norm_layer

        self.conv1 = Conv_BN_ReLU(3, channel_list[0], kernel_size=5, padding=2, stride=2, norm_layer=norm_layer)
        self.conv2 = Conv_BN_ReLU(channel_list[0], channel_list[1], kernel_size=3, padding=1, stride=2, norm_layer=norm_layer)

        self.res_block1 = create_res_block(res_block_list[0], channel_list[1])
        # use 3x3 conv to replace pooling aiming to down sample
        self.down_sample1 = Conv_BN_ReLU(channel_list[1], channel_list[2], kernel_size=3, stride=2, norm_layer=norm_layer)

        self.res_block2 = create_res_block(res_block_list[1], channel_list[2])
        self.down_sample2 = Conv_BN_ReLU(channel_list[2], channel_list[3], kernel_size=3, stride=2, norm_layer=norm_layer)

        self.res_block3 = create_res_block(res_block_list[2], channel_list[3])
        self.down_sample3 = Conv_BN_ReLU(channel_list[3], channel_list[4], kernel_size=3, stride=2, norm_layer=norm_layer)

        self.res_block4 = create_res_block(res_block_list[3], channel_list[4])
        self.down_sample4 = Conv_BN_ReLU(channel_list[4], channel_list[5], kernel_size=3, stride=2, norm_layer=norm_layer)

        self.res_block5 = create_res_block(res_block_list[4], channel_list[5])

        self.feature_map4 = conv1x1(channel_list[5], feature_map_out, stride=1, padding=0)
        self.up_sample4 = Deconv_BN_ReLU(channel_list[5], channel_list[4], kernel_size=2, stride=2, padding=0, norm_layer=norm_layer)
        self.fusion4 = fusion_module(channel_list[4])

        self.feature_map3 = conv1x1(channel_list[4], feature_map_out, stride=1, padding=0)
        self.up_sample3 = Deconv_BN_ReLU(channel_list[4], channel_list[3], kernel_size=2, stride=2, padding=0, norm_layer=norm_layer)
        self.fusion3 = fusion_module(channel_list[3])

        self.feature_map2 = conv1x1(channel_list[3], feature_map_out, stride=1, padding=0)
        self.up_sample2 = Deconv_BN_ReLU(channel_list[3], channel_list[2], kernel_size=2, stride=2, padding=0, norm_layer=norm_layer)
        self.fusion2 = fusion_module(channel_list[2])

        self.feature_map1 = conv1x1(channel_list[2], feature_map_out, stride=1, padding=0)
        self.up_sample1 = Deconv_BN_ReLU(channel_list[2], channel_list[1], kernel_size=2, stride=2, padding=0, norm_layer=norm_layer)

        self.conv3 = Conv_BN_ReLU(channel_list[1], channel_list[1], kernel_size=3, norm_layer=norm_layer)
        self.conv4 = conv1x1(channel_list[1], feature_map_out, stride=1, padding=0)

    def forward(self, x, gt=None):
        feature_map = []
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.res_block1(x)
        x = self.down_sample1(x)

        res_2 = self.res_block2(x)
        x = self.down_sample2(res_2)

        res_3 = self.res_block3(x)
        x = self.down_sample3(res_3)

        res_4 = self.res_block4(x)
        x = self.down_sample4(res_4)

        x = self.res_block5(x)

        feature_map4 = self.feature_map4(x)
        cal = nn.AdaptiveAvgPool2d((feature_map4.shape[2], feature_map4.shape[3]))
        if gt is not None:
            gt_4 = cal(gt)
            loss = self.loss(feature_map4, gt_4)
        x = self.up_sample4(x)
        x = torch.cat((x, res_4), dim=1)
        x = self.fusion4(x)

        feature_map3 = self.feature_map3(x)
        cal = nn.AdaptiveAvgPool2d((feature_map3.shape[2], feature_map3.shape[3]))
        if gt is not None:
            gt_3 = cal(gt)
            loss += self.loss(feature_map3, gt_3)
        x = self.up_sample3(x)
        x = torch.cat((x, res_3), dim=1)
        x = self.fusion3(x)

        feature_map2 = self.feature_map2(x)
        cal = nn.AdaptiveAvgPool2d((feature_map2.shape[2], feature_map2.shape[3]))
        if gt is not None:
            gt_2 = cal(gt)
            loss += self.loss(feature_map2, gt_2)
        x = self.up_sample2(x)
        x = torch.cat((x, res_2), dim=1)
        x = self.fusion2(x)

        feature_map1 = self.feature_map1(x)
        cal = nn.AdaptiveAvgPool2d((feature_map1.shape[2], feature_map1.shape[3]))
        if gt is not None:
            gt_1 = cal(gt)
            loss += self.loss(feature_map1, gt_1)
        x = self.up_sample1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        cal = nn.AdaptiveAvgPool2d((x.shape[2], x.shape[3]))
        if gt is not None:
            gt_0 = cal(gt)
            loss += self.loss(x, gt_0)
        feature_map.append(x)
        feature_map.append(feature_map1)
        feature_map.append(feature_map2)
        feature_map.append(feature_map3)
        feature_map.append(feature_map4)

        if gt is not None:
            return feature_map, loss
        else:
            return feature_map


def haoSegNet(cfg, weight=None):
    cfg = edict(cfg)
    criterion = SegmentationLosses(weight=weight, cuda=torch.cuda.is_available()).build_loss(mode=cfg.loss)
    if cfg.bn == 'bn':
        norm_layer = None
    elif cfg.bn == 'syncbn':
        norm_layer = nn.SyncBatchNorm
    else:
        raise RuntimeError('Unknown bn type, check param model.bn in yaml. Only support bn and syncbn')
    model = HaoSegNet(num_classes=cfg.num_classes, criterion=criterion,
                      channel_list=cfg.channel_list, res_block_list=cfg.res_block_list,
                      norm_layer=norm_layer)
    if cfg.pretrained:
        load_weights(model, cfg.pretrained_model_url)
    return model


if __name__ == "__main__":
    from utils.model_utils import read_yml
    import numpy as np
    cfg = read_yml('/home/lorenzo/PycharmProjects/SemanticSegmentation/Seg_model/cfg/haoSegNet/sparce_ce.yml')
    cfg.model.bn = 'bn'
    model = haoSegNet(cfg.model)
    input_ = torch.randn((1, 3, 512, 512))
    a = np.array([0., 1., 2., 2.]*65536).reshape((512, 512))
    a = a[np.newaxis, :, :]
    target_ = torch.from_numpy(a)

    feature_map, loss = model(input_, target_)
    for feature in feature_map:
        print(feature.shape)
    print(loss)
    # largest_futuremap = model_out[0][0][0]
    # Da_fmap, LL_fmap = SAD_out
    # print(largest_futuremap.shape)