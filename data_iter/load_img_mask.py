import glob
from pathlib import Path
# from torch.utils.data import Dataset
import random


class LoadImgMask(object):
    def __init__(self, args, split="train"):
        self.args = args
        # return train and val - image and mask
        if split == "train":
            if not args.dataset.train_val_split:
                args.dataset.train_val_split_ratio = 0
            self.train_path, self.val_path = self._get_path(self.args.dataset.train_dir,
                                                            args.dataset.train_val_split_ratio,
                                                            args.SEED, split)

            if args.ddp.LOCAL_RANK in [0, -1]:
                num_train_imgs = len(self.train_path)
                num_val_imgs = len(self.val_path)
                num_total_imgs = num_train_imgs + num_val_imgs
                print("Dataset statistics:")
                print("  ------------------------------")
                print("  subset     | # images")
                print("  ------------------------------")
                print("  total      | {:5d} ".format(num_total_imgs))
                print("  train      | {:5d} ".format(num_train_imgs))
                print("  val        | {:5d} ".format(num_val_imgs))
                print("  ------------------------------")
        elif split == "val":
            self.val_path = self._get_path(self.args.dataset.train_dir,
                                           args.dataset.train_val_split_ratio,
                                           args.SEED, split)
        elif split == "test":
            self.test_path = self._get_path(self.args.dataset.train_dir,
                                            args.dataset.train_val_split_ratio,
                                            args.SEED, split)
        else:
            raise RuntimeError("split must be train/val/test")

    def _get_path(self, img_dir, val_factor, seed, split):
        img_path = Path(img_dir) / 'images' / split
        mask_path = Path(img_dir) / 'masks' / split
        self._check_path([img_path, mask_path])
        img_paths = self._get_imgs(img_path)
        if split == "train":
            length = len(img_paths)
            val_size = int(length * val_factor)
            random.seed(seed)
            random.shuffle(img_paths)
            val_img = img_paths[:val_size]
            train_img = img_paths[val_size:]
            val_path = self._get_responding_mask(val_img, mask_path)
            train_path = self._get_responding_mask(train_img, mask_path)
            return train_path, val_path
        elif split == "val":
            val_path = self._get_responding_mask(img_paths, mask_path)
            return val_path
        elif split == "test":
            test_path = self._get_responding_mask(img_paths, mask_path)
            return test_path

    def _get_responding_mask(self, img_list, train_mask_path):
        extensions = ['.tif', '.TIF', '.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']
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

    def _get_imgs(self, img_dir):
        extensions = ['tif', 'TIF', 'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        img_paths = []
        for ext in extensions:
            try:
                img_paths += glob.glob(str(img_dir / "*.{}".format(ext)))
            except:
                raise RuntimeError("folder not found")
        return img_paths

    def _check_path(self, paths_list):
        for path in paths_list:
            if not path.is_dir():
                raise RuntimeError("'{}' is not a dir".format(path))


if __name__ == "__main__":
    from utils.model_utils import read_yml
    cfg = read_yml('../cfg/UNet/cityscapes_ce_lr1e3_bs32.yml')
    dataset = LoadImgMask(cfg, cfg.dataset.train_dir)
    print('')
