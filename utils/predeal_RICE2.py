import os
import glob
import random
from pathlib import Path
import shutil
from PIL import Image
import numpy as np


def get_imgs(img_dir):
    extensions = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    img_paths = []
    for ext in extensions:
        try:
            img_paths += glob.glob(str(img_dir / "*.{}".format(ext)))
        except:
            raise RuntimeError("folder not found")
    return img_paths


def get_responding_mask(img_list, train_mask_path):
    extensions = ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']
    img_paths = []
    for i, path in enumerate(img_list):
        for ext in extensions:
            mask_path = train_mask_path / "".join((Path(path).stem, ext))
            if mask_path.exists():
                path_dict = {'image': path, 'label': str(mask_path)}
                img_paths.append(path_dict)
                break
        if len(img_paths) != (i+1):
            raise RuntimeError("'{}' mask not found".format(path))
    return img_paths


def train_test_split(root_path):
    img_path = Path(root_path) / 'cloud'
    mask_path = Path(root_path) / 'mask'
    img_paths = get_imgs(img_path)
    length = len(img_paths)
    test_size = int(length * 0.1)
    random.seed(666)
    random.shuffle(img_paths)
    test_img = img_paths[:test_size]
    train_img = img_paths[test_size:]
    test_path = get_responding_mask(test_img, mask_path)
    train_path = get_responding_mask(train_img, mask_path)
    return train_path, test_path


def create_my_dataset(path_dict, save_path, split):
    for sample in path_dict:
        img_path, mask_path = sample['image'], sample['label']
        img_name = os.path.basename(img_path)
        save_img_path = os.path.join(save_path, 'images', split, img_name)
        save_color_path = os.path.join(save_path, 'color', split, img_name)
        save_mask_path = os.path.join(save_path, 'masks', split, img_name)
        mask = np.array(Image.open(save_color_path))
        mask_sum = np.sum(mask, axis=2)
        mask[mask_sum == 765] = 1    # cloud label: 1
        mask[mask_sum == 255] = 2    # cloud shadow label: 2
        mask[mask_sum == 510] = 3    # cirrus label: 3
        mask_img = Image.fromarray(mask)
        mask_img.save(save_mask_path)
        shutil.copyfile(img_path, save_img_path)
        shutil.copyfile(mask_path, save_color_path)


if __name__ == "__main__":
    root_path = "/mnt2/sjh/seg_data/RICE_DATASET/RICE2"
    save_path = "/mnt2/sjh/seg_data/myRICE"
    train_path, test_path = train_test_split(root_path)
    create_my_dataset(train_path, save_path, split='train')
    create_my_dataset(test_path, save_path, split='test')
