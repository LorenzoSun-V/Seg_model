import os
import math
import pathlib

import cv2
import logging
import json
from pprint import pprint
import logzero
import yaml
import time
import numpy as np
from logzero import logger as log
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
from easydict import EasyDict as edict
from pathlib import Path, PosixPath
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools
# from utils.loss import *
import warnings


def set_seed(local_rank, seed=666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    if local_rank in [-1, 0]:
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')


def checkfolder(paths):
    print(type(paths))
    if isinstance(paths, str):
        if not Path(paths).is_dir():
            os.mkdir(paths)
            log.info("Created new directory in %s" % paths)

    if isinstance(paths, PosixPath) or isinstance(paths, pathlib.WindowsPath):
        if not Path(paths).is_dir():
            paths.mkdir(parents=True)
            # Path.mkdir(paths)
            log.info("Created new directory in %s" % paths)


def checkbntype(cfg):
    if cfg.USE_DDP:
        if cfg.model.bn == "bn":
            log.info('DDP mode is suggested to use syncbn type. Have modified automatically.')
    else:
        if cfg.model.bn == "syncbn":
            log.info('Common mode is suggested to use bn type. Have modified automatically.')
            cfg.model.bn = "bn"
    return cfg


def read_yml(yml_file):
    with open(yml_file, encoding='utf-8') as f:
        cfg = edict(yaml.safe_load(f))
    return cfg


def train_val_split(train_img_path, train_mask_path, val_factor):
    log.info('=> train val split ')

    train_img_path = Path(train_img_path)
    train_mask_path = Path(train_mask_path)
    img_paths = [path for path in train_img_path.iterdir() if path.suffix == '.jpg' or path.suffix == '.png']
    length = len(img_paths)
    val_size = int(length*val_factor)
    random.shuffle(img_paths)
    val_img = img_paths[:val_size]
    train_img = img_paths[val_size:]
    val_mask = [train_mask_path / "".join((path.stem, ".png")) for path in val_img]
    train_mask = [train_mask_path / "".join((path.stem, ".png")) for path in train_img]
    return train_img, train_mask, val_img, val_mask


def cal_lr_lambda(epochs, warmup_cos_decay):
    # learning rate warmup and cosine decay
    t = warmup_cos_decay
    lambda1 = lambda epoch: ((epoch+1) / t) if (epoch+1) <= t else 0.1 \
        if 0.5 * (1+math.cos(math.pi*(epoch+1-t) / (epochs-t))) < 0.1 else 0.5 * (1+math.cos(math.pi*(epoch+1-t)/(epochs-t)))
    return lambda1


def create_log_dir(cfg):
    # print train set args, create pth and log dir
    local_time = time.localtime()
    if cfg.LOG_DIR == "":
        # create folder saving log files and pth files adaptively via model_exp & model_name & task_name
        # path like:  /model_exp/model_name/task_name/2021-09-28_13-47-26
        pth_path = Path(cfg.dataset.model_exp)
        pth_path = pth_path / cfg.dataset.task_name / cfg.model.model_name / time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        cfg.LOG_DIR = str(pth_path)
    else:
        # create folder via LOG_DIR
        pth_path = Path(cfg.LOG_DIR)
    checkfolder(Path(pth_path))

    log_path = str(pth_path / "log.txt")
    logzero.logfile(log_path, maxBytes=1e6, backupCount=3)
    log.info("################################################ NEW LOG")
    pprint(cfg)
    fs = open(pth_path / 'train_ops.json', "w", encoding='utf-8')
    json.dump(cfg, fs, ensure_ascii=False, indent=1)
    fs.close()

    return cfg, log_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_fscore(y_pred, y_true):
    eps = torch.FloatTensor([1e-7])
    beta = torch.FloatTensor([1])

    true_positive = (y_pred * y_true).sum(dim=0)
    precision = true_positive.div(y_pred.sum(dim=0).add(eps)) ##p = tp / (tp + fp + eps)
    recall = true_positive.div(y_true.sum(dim=0).add(eps)) ##r = tp / (tp + fn + eps)
    micro_f1 = torch.mean((precision*recall).div(precision.mul(beta).mul(beta) + recall + eps).mul(1 + beta).mul(1+beta))

    return precision, recall, micro_f1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def load_weights(model: nn.Module, model_url: str):
    state_dict = torch.load(model_url, map_location=lambda storage, loc:storage)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(f'Error: The pretrained weights from "{model_url}" cannot be loaded')
        exit(0)
    else:
        print(f'Successfully loaded weights from {model_url}')
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                f'due to unmatched keys or layer size: {discarded_layers}')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.rc('font', family='sans-serif', size='4.5')   # 设置字体样式、大小
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(200, 200))
    plt.rcParams['figure.dpi'] = 200 #分辨率
    cm = cm.astype(np.uint64)
    if normalize:
    # 按行进行归一化
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if cm[i, j] == 0:
    #             cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    plt.title('Confusion matrix')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='GT',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            ax.text(j, i, format(cm[i, j], fmt)+'%' if normalize else format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    cm = np.array([[30458,0,0,110,6,37,8],
                    [0,4217,0,3,0,0,0],
                     [2,0,3103,0,0,0,0],
                     [3,2,4,12262,6,88,1],
                     [3,1,0,25,15247,0,0],
                     [0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0]])
    label = {'0': 'bank_staff_vest', '1': 'cleaner', '2': 'money_staff', '3': 'person', '4': 'security_staff',
                '5': 'bank_staff_shirt', '6': 'bank_staff_coat'}
    label_class = [label[i] for i in label]
    plot_confusion_matrix(cm, label_class)




