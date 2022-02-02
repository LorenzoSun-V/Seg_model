import os
import cv2
import numpy as np
import glob


image_path = "/mnt/shy/sjh/Seg_data/iccv09Data/images"
label_path = "/mnt/shy/sjh/Seg_data/iccv09Data/labels"
mask_path = "/mnt/shy/sjh/Seg_data/iccv09Data/masks"
vis_path = "/mnt/shy/sjh/Seg_data/iccv09Data/masks_color"
palette = [[0, 0, 0], [255, 255, 255], [255, 0, 0],
           [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
           [250, 170, 30], [220, 220,  0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255,  0,  0], [0,  0, 142], [0,  0, 70], [0, 60, 100], [0, 80, 100], [0,  0, 230], [119, 11, 32]]


def addmask2img(img, mask):
    color_area = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(np.max(mask)+1):
        color_area[mask==i] = palette[i]
    color_seg = color_area
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    # img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)
    # mask = np.expand_dims(mask, axis=2)
    # img2 = mask + img
    return img


def main():
    label_dirs = glob.glob(os.path.join(label_path, "*.regions.txt"))
    for label_dir in label_dirs:
        image_name = os.path.basename(label_dir).split('.')[0] + '.jpg'
        img_original = cv2.imread(os.path.join(image_path, image_name))
        with open(label_dir, 'r') as f:
            mask = []
            for i, line in enumerate(f.readlines()):
                line = line.strip('\n').split(' ')
                new_line = []
                for pix in line:
                    pix = int(pix)
                    if pix < 0:
                        pix = 0
                    new_line.append(pix)
                if i == 0:
                    mask = np.array(new_line)
                else:
                    mask_stack = np.array(new_line)
                    mask = np.row_stack((mask, mask_stack))
        color_mask = addmask2img(img_original, mask)
        cv2.imwrite(os.path.join(mask_path, image_name.replace('jpg', 'png')), mask)
        cv2.imwrite(os.path.join(vis_path, image_name), color_mask)


if __name__ == "__main__":
    mask = cv2.imread("/mnt/shy/sjh/Seg_data/iccv09Data/masks/0000047.png")
    print(mask)