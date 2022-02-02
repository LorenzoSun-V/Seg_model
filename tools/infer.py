import torch
from torch.utils.data import DataLoader
import shutil
import argparse
from tqdm import tqdm
from utils.model_utils import *
from utils.metrics import *
from models.yanghao_model.haoSegNet import haoSegNet
from models.UNet.unet_model import unet
from models.deeplab.deeplab import deeplab
from data_iter.load_img_mask import LoadImgMask
from data_iter.dataset_iter import DataIter, create_dataloader
from utils.model_utils import plot_confusion_matrix


label_cityscapes = {'0': 'road', '1': 'sidewalk', '2': 'building', '3': 'wall', '4': 'fence', '5': 'pole',
                    '6': 'traffic light', '7': 'traffic sign', '8': 'vegetation', '9': 'terrain', '10': 'sky',
                    '11': 'person', '12': 'rider', '13': 'car', '14': 'truck', '15': 'bus', '16': 'train',
                    '17': 'motorcycle', '18': 'bicycle'}
label_sparcs = {'0': 'land', '1': 'cloud', '2': 'shadow'}


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/UNet/sparcs_ce.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


def eval_softmax(model, loader, num_classes, label):
    model.eval()
    evaluator = Evaluator(num_classes)
    evaluator.reset()

    with torch.no_grad():
        for batch_id, sample in enumerate(tqdm(loader)):
            img, mask = sample['image'], sample['label']
            img = img.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)

            pred_mask = model(img)
            pred = pred_mask.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(mask, pred)

    Acc = evaluator.Pixel_Accuracy(evaluator.confusion_matrix)
    Acc_class = evaluator.Pixel_Accuracy_Class(evaluator.confusion_matrix)
    mIoU = evaluator.Mean_Intersection_over_Union(evaluator.confusion_matrix)
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union(evaluator.confusion_matrix)
    label_class = [label[i] for i in label]
    plot_confusion_matrix(evaluator.confusion_matrix, classes=label_class, normalize=False, title='confusion matrix')
    print(
        f"Test | Acc: {Acc:.4f} | Acc_class: {Acc_class:.4f} | mIoU: {mIoU:.4f} | FWIoU: {FWIoU:.4f}\n")


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    if len(cfg.test.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu
    else:
        raise RuntimeError("please use single gpu, change yaml file param: test.gpu")

    dataset = LoadImgMask(cfg, split='test')
    test_dataiter = DataIter(cfg.aug, dataset.test_path, split='test')
    test_dataloader = DataLoader(test_dataiter,
                                 batch_size=cfg.test.batch_size,
                                 pin_memory=True,
                                 shuffle=False)
    model_ = eval(f"{cfg.model.model_name}({cfg.model})")
    load_weights(model_, cfg.test.model_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()
    task_name = cfg.dataset.task_name
    if task_name == 'cityscapes':
        label = label_cityscapes
    elif task_name == 'sparcs':
        label = label_sparcs

    eval_softmax(model_, test_dataloader, cfg.model.num_classes, label)


