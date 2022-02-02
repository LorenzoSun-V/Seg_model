import math
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
from data_iter.load_img_mask import LoadImgMask
from data_iter.dataset_iter import DataIter, create_dataloader
from utils.model_utils import read_yml


def arg_define():
    parser = argparse.ArgumentParser(description='Seg model train')
    parser.add_argument('--yml', type=str, default='../cfg/UNet/sparcs_ce.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


def calculate_weights_labels(cfg, dataloader):
    num_classes = cfg.model.num_classes
    z = np.zeros((num_classes,))
    tqdm_batch = tqdm(dataloader)
    for (img, label) in tqdm_batch:
        y = label.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    # print(z)
    total_frequency = np.sum(z)
    # max_frequency = np.max(z)
    # mu = 1.0 / (total_frequency / max_frequency)
    class_weights = []
    for frequency in z:
        # class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weight = frequency / total_frequency
        # class_weight = math.log(mu * total_frequency / frequency)
        class_weights.append(class_weight)
    ret = np.array(class_weights) * 19
    class_weights_path = Path(cfg.dataset.model_exp)
    class_weights_path = str(class_weights_path / cfg.dataset.task_name / "classes_weights.npy")
    np.save(class_weights_path, ret)
    print(ret)
    return ret


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    dataset = LoadImgMask(cfg, split='test')
    total_dataiter = DataIter(cfg.aug, dataset.test_path, split='test')
    test_dataloader = DataLoader(test_dataiter,
                                 batch_size=cfg.test.batch_size,
                                 pin_memory=True,
                                 shuffle=False)
    calculate_weights_labels(cfg, total_dataloader)
