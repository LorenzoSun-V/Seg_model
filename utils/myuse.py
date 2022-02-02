import os
import glob
import numpy as np
import cv2
from PIL import Image
from skimage import io


pre_masks_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome/pre_masks"
save_masks_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome/masks"

mask_dirs = glob.glob(os.path.join(pre_masks_path, "*.tif"))
for mask_dir in mask_dirs:
    image_name = os.path.basename(mask_dir).split('_')[0]
    save_path = os.path.join(save_masks_path, f"{image_name}.tif")
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    mask[mask == 255] = 1
    mask[mask == 192] = 1
    mask[mask == 0] = 255
    mask[mask == 128] = 0
    mask[mask == 64] = 2
    io.imsave(save_path, mask)
    # print(np.unique(mask))
    # cv2.namedWindow("a", 0)
    # cv2.resizeWindow("a", 2000, 2000)
    # cv2.imshow("a", mask)
    # cv2.waitKey(0)
