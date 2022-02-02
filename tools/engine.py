import time
import torch
from tqdm import tqdm
import numpy as np
from logzero import logger as log
from utils.model_utils import AverageMeter
import torch.distributed as dist


def average_gradients(model):
    """
    Reduce the gradients from all processes so that
    process with rank 0 has the averaged gradients
    """
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size


def reduce_tensor(inp):
    """
    Reduce the results from all processes so that
    process with rank 0 has the averaged results
    """
    world_size = float(dist.get_world_size())
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduce_inp = inp
        dist.reduce(reduce_inp, dst=0)
    return reduce_inp / world_size


def train_one_epoch(loader, USE_DDP, evaluator, model, optimizer, device, print_freq, epoch, epochs, local_rank):
    model.train()
    evaluator.reset()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, sample in enumerate(loader):
        img, mask = sample['image'], sample['label']
        data_time.update(time.time() - end)
        img = img.to(device=device, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)
        # calculate loss
        pred_mask, loss = model(img, mask)
        # calculate metrics
        pred = pred_mask.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(mask, pred)
        Acc_minibatch = evaluator.Pixel_Accuracy(evaluator.confusion_matrix_minibatch)
        mIoU_minibatch = evaluator.Mean_Intersection_over_Union(evaluator.confusion_matrix_minibatch)

        if USE_DDP:
            reduced_loss = reduce_tensor(loss)
            reduced_acc = reduce_tensor(Acc_minibatch)
            reduced_mIoU = reduce_tensor(mIoU_minibatch)
        else:
            reduced_loss = loss
            reduced_acc = Acc_minibatch
            reduced_mIoU = mIoU_minibatch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(reduced_loss.data.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx % print_freq == 0) and (local_rank in [-1, 0]):
            log.info(f"TRAIN | Epoch: [{epoch}/{epochs}][{batch_idx}/{len(loader)}] | Time: {data_time.val:.3f}/{batch_time.val:.3f} | Loss: {losses.val:.3f} | Acc: {reduced_acc:.3f} | mIoU: {reduced_mIoU:.3f}")

    Acc = evaluator.Pixel_Accuracy(evaluator.confusion_matrix)
    Acc_class = evaluator.Pixel_Accuracy_Class(evaluator.confusion_matrix)
    mIoU = evaluator.Mean_Intersection_over_Union(evaluator.confusion_matrix)
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union(evaluator.confusion_matrix)
    if USE_DDP:
        reduced_acc = reduce_tensor(Acc)
        reduced_acc_class = reduce_tensor(Acc_class)
        reduced_mIoU = reduce_tensor(mIoU)
        reduced_FWIoU = reduce_tensor(FWIoU)
    else:
        reduced_acc = Acc
        reduced_acc_class = Acc_class
        reduced_mIoU = mIoU
        reduced_FWIoU = FWIoU
    if local_rank in [-1, 0]:
        print(type(losses.avg), type(Acc), type(Acc_class), type(mIoU))
        log.info(
            f"TRAIN | Epoch: [{epoch}/{epochs}] | Time: {batch_time.sum:.3f} | Loss: {losses.avg:.3f} | Acc: {reduced_acc:.3f} | Acc_class: {reduced_acc_class:.3f} | mIoU: {reduced_mIoU:.3f} | FWIoU: {reduced_FWIoU:.3f} \n")

    return losses.avg, reduced_acc, reduced_acc_class, reduced_mIoU, reduced_FWIoU


def val_one_epoch(loader, USE_DDP, evaluator, model, device, local_rank):
    model.eval()
    evaluator.reset()
    losses = AverageMeter()

    with torch.no_grad():
        for sample in tqdm(loader, total=len(loader)):
            img, mask = sample['image'], sample['label']
            img = img.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            # calculate loss
            pred_mask, loss = model(img, mask)

            # calculate metrics
            pred = pred_mask.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(mask, pred)

            if USE_DDP:
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            losses.update(reduced_loss)

    Acc = evaluator.Pixel_Accuracy(evaluator.confusion_matrix)
    Acc_class = evaluator.Pixel_Accuracy_Class(evaluator.confusion_matrix)
    mIoU = evaluator.Mean_Intersection_over_Union(evaluator.confusion_matrix)
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union(evaluator.confusion_matrix)
    if USE_DDP:
        reduced_acc = reduce_tensor(Acc)
        reduced_acc_class = reduce_tensor(Acc_class)
        reduced_mIoU = reduce_tensor(mIoU)
        reduced_FWIoU = reduce_tensor(FWIoU)
    else:
        reduced_acc = Acc
        reduced_acc_class = Acc_class
        reduced_mIoU = mIoU
        reduced_FWIoU = FWIoU
    if local_rank in [-1, 0]:
        log.info(f"Val | Loss: {losses.avg:.4f} | Acc: {reduced_acc:.4f} | Acc_class: {reduced_acc_class:.4f} | mIoU: {reduced_mIoU:.4f} | FWIoU: {reduced_FWIoU:.4f}\n")
    return losses.avg, reduced_acc, reduced_mIoU
