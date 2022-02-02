import os
import cv2
import imageio
import numpy as np
import glob


# image_path = "/mnt2/sjh/seg_data/myCityscapes/images/val"
# mask_path = "/mnt2/sjh/seg_data/myCityscapes/masks/val"
# vis_path = "/mnt2/sjh/seg_data/myCityscapes/vis/val"
# palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
#            [250, 170, 30], [220, 220,  0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
#            [255,  0,  0], [0,  0, 142], [0,  0, 70], [0, 60, 100], [0, 80, 100], [0,  0, 230], [119, 11, 32]]

# image_path = "/home/user/Downloads/images"
# mask_path = "/home/user/Downloads/new_labels"
# vis_path = "/home/user/Downloads/vis"

image_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut/false_color"
mask_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut/masks"
vis_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut/vis"
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 255], [0, 0, 255], [0, 60, 100]]


def addmask2img(img, mask):
    color_area = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_area[mask==255] = [0,0,0]
    for i in range(4):
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
    img_dirs = glob.glob(os.path.join(image_path, '*.tif'))
    img_dirs.sort()
    mask_dirs = glob.glob(os.path.join(mask_path, '*.tif'))
    mask_dirs.sort()
    for img_dir, mask_dir in zip(img_dirs, mask_dirs):
        print(img_dir, mask_dir)
        img = cv2.imread(img_dir)
        mask = cv2.imread(mask_dir, -1)
        new_img = addmask2img(img, mask)
        save_path = os.path.join(vis_path, os.path.basename(img_dir))
        cv2.imwrite(save_path, new_img)


if __name__ == "__main__":
    # main()
    from skimage import io
    mask = io.imread("/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myCityspaces/masks/train/aachen_000000_000019.png")
    print(mask.shape)
    mask_a = np.unique(mask)
    print(mask_a)