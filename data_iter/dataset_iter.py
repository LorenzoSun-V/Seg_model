from PIL import Image
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from logzero import logger as log
import numpy as np
from torchvision import transforms
import data_iter.zoey_transforms as zoey


class DataIter(Dataset):
    def __init__(self, args, dataset, split="train"):
        super(DataIter, self).__init__()
        self.args = args
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path_dict = self.dataset[index]
        img_path, mask_path = path_dict['image'], path_dict['label']
        img_original = Image.open(img_path)
        gt_original = Image.open(mask_path)
        sample = {'image': img_original, 'label': gt_original}
        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_test(sample)
        else:
            raise RuntimeError("data_iter type is train/val/test")

    def transform_train(self, sample):
        transform_list = []
        if self.args.CROP:
            transform_list.append(
                zoey.RandomScaleCrop(base_size=self.args.HW[0], crop_size=self.args.HW[0], fill=255))
        if self.args.is_aug:
            if self.args.HFLIP:
                transform_list.append(zoey.RandomHorizontalFlip())
            if self.args.VFLIP:
                transform_list.append(zoey.RandomVerticalFlip())
            if self.args.Brightness > 0:
                transform_list.append(zoey.Brightness(brightness=self.args.Brightness))
            if self.args.BLUR:
                transform_list.append(zoey.RandomGaussianBlur())
        transform_list.append(zoey.Normalize(mean=self.args.Mean, std=self.args.Std))
        transform_list.append(zoey.ToTensor())
        composed_transforms = transforms.Compose(transform_list)

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            zoey.FixScaleCrop(crop_size=self.args.HW[0]),
            zoey.Normalize(mean=self.args.Mean, std=self.args.Std),
            zoey.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            zoey.FixedResize(size=self.args.HW[0]),
            zoey.Normalize(mean=self.args.Mean, std=self.args.Std),
            zoey.ToTensor()
        ])

        return composed_transforms(sample)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, dataiter, local_rank):
    assert cfg.train.batch_size % cfg.ddp.NPROCS == 0, "batch size % nprocs is not 0"
    batch_size = int(cfg.train.batch_size / cfg.ddp.NPROCS)
    # print("****", cfg.ddp.LOCAL_RANK, batch_size)
    nw = min([os.cpu_count() // cfg.ddp.WORLD_SIZE, batch_size if batch_size > 1 else 0, 8])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataiter) if local_rank != -1 else None
    dataloader = DataLoaderX(dataiter,
                             batch_size=batch_size,
                             num_workers=nw,
                             sampler=sampler,
                             pin_memory=True)
    return dataloader
