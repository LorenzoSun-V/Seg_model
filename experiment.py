import cv2
import numpy as np
from PIL import Image
from skimage import io
import os
import glob
from pathlib import Path
import math


def cal_points_distance(coord1, coord2):
    x1, y1, x2, y2 = coord1[0], coord1[1], coord2[0], coord2[1]
    d = (x1-x2) ** 2 + (y1-y2) ** 2
    return math.sqrt(d)


image_root_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome/pre_images"
false_root_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome/images_false_color"
mask_root_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome/pre_masks"

mask_dirs = glob.glob(str(Path(mask_root_path) / "*.tif"))
for mask_dir in mask_dirs:
    image_name = os.path.basename(mask_dir).split('_')[0]
    false_color_path = os.path.join(false_root_path, f"{image_name}.tif")
    image_path = os.path.join(image_root_path, f"{image_name}.tif")
    mask = cv2.imread(mask_dir)
    false_color_image = cv2.imread(false_color_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = contours[0][:, 0]
    x_max = np.max(new_contours[:, 0])
    x_min = np.min(new_contours[:, 0])
    y_max = np.max(new_contours[:, 1])
    y_min = np.min(new_contours[:, 1])
    x_max_coord = new_contours[np.argwhere(new_contours[:, 0] == x_max)][0, 0]
    x_min_coord = new_contours[np.argwhere(new_contours[:, 0] == x_min)][0, 0]
    y_max_coord = new_contours[np.argwhere(new_contours[:, 1] == y_max)][0, 0]
    y_min_coord = new_contours[np.argwhere(new_contours[:, 1] == y_min)][0, 0]
    # print(x_min, x_max, y_min, y_max)
    my_coord = np.array([x_min_coord, y_max_coord, x_max_coord, y_min_coord], np.float32)
    my_coord = my_coord.reshape((-1, 1, 2))
    d = int((cal_points_distance(x_min_coord, y_max_coord) + cal_points_distance(y_max_coord, x_max_coord) +
         cal_points_distance(x_max_coord, y_min_coord) + cal_points_distance(y_min_coord, x_min_coord)) / 4)
    # mask_new = cv2.polylines(mask, [my_coord], True, (0, 255, 255), 5)
    coord_new = np.array([[0, d], [d, d], [d, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(my_coord, coord_new)
    mask_new = cv2.warpPerspective(mask, M, (d, d))[0]
    a = np.unique(mask)
    false_color_image_new = cv2.warpPerspective(false_color_image, M, (d, d))
    # cv2.namedWindow("a", 0)
    # cv2.namedWindow("b", 0)
    # cv2.resizeWindow("a", 1000, 1000)
    # cv2.resizeWindow("b", 1000, 1000)
    # cv2.imshow("a", mask_new)
    # cv2.imshow("b", false_color_image_new)
    # cv2.waitKey(0)

    # cv2.drawContours(mask, contours, -1, (0, 0, 255), 3)
    # cv2.namedWindow("a", 0)
    # cv2.resizeWindow("a", 2000, 2000)
    # cv2.imshow("a", mask)
    # cv2.waitKey(0)
    # print(contours)

