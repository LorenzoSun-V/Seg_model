import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_minibatch = np.zeros((self.num_class,) * 2)
        self.epsilon = 1e-10

    def Pixel_Accuracy(self, confusion_matrix):
        Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        return torch.from_numpy(np.array(Acc)).cuda()

    def Pixel_Accuracy_Class(self, confusion_matrix):
        Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return torch.from_numpy(np.array(Acc)).cuda()

    def Mean_Intersection_over_Union(self, confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return torch.from_numpy(np.array(MIoU)).cuda()

    def Frequency_Weighted_Intersection_over_Union(self, confusion_matrix):
        freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
        iu = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return torch.from_numpy(np.array(FWIoU)).cuda()

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # print("gt:", gt_image.shape)
        # print("pre:", pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix_minibatch = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += self.confusion_matrix_minibatch

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_minibatch = np.zeros((self.num_class,) * 2)


if __name__ == "__main__":
    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(np.array([[0,2,1],[2,1,0],[1,0,4]]).reshape(3,3), np.array([[1,0,2],[0,2,1],[2,1,4]]))
    Acc_minibatch = evaluator.Pixel_Accuracy(evaluator.confusion_matrix_minibatch)
    mIoU_minibatch = evaluator.Mean_Intersection_over_Union(evaluator.confusion_matrix_minibatch)
    print(Acc_minibatch, mIoU_minibatch)