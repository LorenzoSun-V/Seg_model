import cv2
import numpy as np
import torch
from models.UNet.unet_model import unet
from models.deeplab.deeplab import deeplab
import argparse
from utils.model_utils import *
import glob
from utils.stanford_data import addmask2img


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/UNet/sparcs_ce.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    if len(cfg.test.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu
    else:
        raise RuntimeError("please use single gpu, change yaml file param: test.gpu")
    model_ = eval(f"{cfg.model.model_name}({cfg.model})")
    load_weights(model_, cfg.test.model_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()
    img_dirs = glob.glob(os.path.join(cfg.test.test_dir, "*.png"))
    for img_dir in img_dirs:
        img_name = os.path.basename(img_dir)
        img = cv2.imread(img_dir)
        original_img = img
        original_h, original_w = img.shape[:2]
        img = cv2.resize(img, (cfg.aug.HW[1], cfg.aug.HW[0]), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        img -= cfg.aug.Mean
        img /= cfg.aug.Std
        img_trans = img.transpose(2, 0, 1).astype(np.float32)
        img_trans = img_trans[np.newaxis, :, :, :]
        img_tensor = torch.from_numpy(img_trans)
        inputs = img_tensor.to(device=device, non_blocking=True)
        logits = model_(inputs)
        logits = torch.softmax(logits, dim=1)
        preds = torch.topk(logits, 1, dim=1)[1]
        preds = preds.squeeze().cpu().detach().numpy()
        preds = cv2.resize(preds, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        color_mask = addmask2img(original_img, preds)
        cv2.imwrite(os.path.join(cfg.test.test_save_dir, img_name), preds)
        cv2.imwrite(os.path.join(cfg.test.color_save_dir, img_name), color_mask)
        print(preds.shape)
