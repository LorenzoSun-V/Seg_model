from utils.model_utils import plot_confusion_matrix
import numpy as np


a = np.ones((19, 19))
a[2, 1] = 0
a[1, 2] = 3
label_cityscapes = {'0': 'road', '1': 'sidewalk', '2': 'building', '3': 'wall', '4': 'fence', '5': 'pole',
                    '6': 'traffic light', '7': 'traffic sign', '8': 'vegetation', '9': 'terrain', '10': 'sky',
                    '11': 'person', '12': 'rider', '13': 'car', '14': 'truck', '15': 'bus', '16': 'train',
                    '17': 'motorcycle', '18': 'bicycle'}
label = [label_cityscapes[i] for i in label_cityscapes]
plot_confusion_matrix(a, label)
