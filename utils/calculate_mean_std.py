import glob
import os
from tqdm import tqdm
import torch
import time
import argparse
from data_iter.load_img_mask import LoadImgMask
from data_iter.dataset_iter import DataIter, create_dataloader
from utils.model_utils import read_yml


def arg_define():
    parser = argparse.ArgumentParser(description='Seg model train')
    parser.add_argument('--yml', type=str, default='../cfg/UNet/cityscapes_ce_lr1e3_bs32.yml', help='path of cfg file')
    args = parser.parse_args()
    return args

from PIL import Image
import numpy as np
img_dirs = glob.glob(os.path.join("/mnt2/sjh/seg_data/mySPARCS/images/train", "*.png"))
mean = np.zeros(3)
std = np.zeros(3)
for img_dir in tqdm(img_dirs):
    img = np.array(Image.open(img_dir)) / 255.0
    for i in range(3):
        mean[i] += np.mean(img[:, :, i])
        std[i] += np.std(img[:, :, i])
mean_total = mean / len(img_dirs)
std_total = std / len(img_dirs)
print(mean_total, std_total)


# def calculate_mean_std(cfg, total_dataiter, n_channels):
#     total_dataloader = create_dataloader(cfg, total_dataiter, -1)
#     start = time.time()
#     mean = torch.zeros(n_channels)
#     std = torch.zeros(n_channels)
#     print('=> Computing mean and std ..')
#     for sample in tqdm(total_dataloader):
#         images = sample['image']
#         for i in range(n_channels):
#             mean[i] += images[:, i, :, :].mean()
#             std[i] += images[:, i, :, :].std()
#     mean.div_(total_dataiter.__len__())
#     std.div_(total_dataiter.__len__())
#     print(mean, std)
#
#     print(f"time elapsed: {time.time() - start}")
#
#
# if __name__ == "__main__":
#     args = arg_define()
#     cfg = read_yml(args.yml)
#     cfg.dataset.train_val_split = False
#     dataset = LoadImgMask(cfg)
#     total_dataiter = DataIter(cfg, dataset.train_path, split='val')
#     calculate_mean_std(cfg, total_dataiter, n_channels=3)

