import os
import glob
from skimage import io
from PIL import Image
import numpy as np
from pathlib import Path
from threading import Thread
import cv2
import shutil
from tqdm import tqdm
import random
from collections import Counter


class DataProcessor:
    def __init__(self, root_path, save_root_path, save_cut_root_path):
        self.root_path = root_path
        self.save_root_path = save_root_path
        self.save_cut_root_path = save_cut_root_path

    def _search_tif(self, img_dirs):
        # Biome dataset: B2_Blue, B3_Green, B4_Red, B5_Nir
        # tif_dict is like: {"aa": ["a.TIF", "b.TIF", "c.TIF", "mask.TIF"]}
        tif_dict = dict()
        bands = ["B2.TIF", "B3.TIF", "B4.TIF", "B5.TIF"]
        labels_path = glob.glob(str(Path("/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_labels") / "*.tif"))

        for image_dir in img_dirs:
            image_name = os.path.basename(image_dir)
            rgbn_path = []
            tiff_paths = glob.glob(str(Path(image_dir) / "*.TIF"))
            for tiff_path in tiff_paths:
                for band in bands:
                    if band in tiff_path:
                        rgbn_path.append(tiff_path)
            for label_path in labels_path:
                if image_name in label_path:
                    rgbn_path.append(label_path)
            tif_dict[image_name] = rgbn_path

        return tif_dict

    def _transfer16bit28bit(self, img_16bit):
        min_16bit = np.min(img_16bit)
        max_16bit = np.max(img_16bit)
        # print(f'16bit dynamic range: {min_16bit} - {max_16bit}')
        img_8bit = np.array(255 * ((img_16bit - min_16bit) / (max_16bit - min_16bit)), dtype=np.uint8)
        # print(f'8bit dynamic range: {np.min(img_8bit)} - {np.max(img_8bit)}')
        return img_8bit

    def _biome(self, tif_dict):
        for current_img in tif_dict:
            for each_band in tif_dict[current_img]:
                if "B2.TIF" in each_band:
                    blue = io.imread(each_band)
                elif "B3.TIF" in each_band:
                    green = io.imread(each_band)
                elif "B4.TIF" in each_band:
                    red = io.imread(each_band)
                elif "B5.TIF" in each_band:
                    nir = io.imread(each_band)
                elif "mask" in each_band:
                    mask = io.imread(each_band)

            # turn 16-bit data to 8-bit data
            img = np.zeros((red.shape[0], red.shape[1], 4))
            img[:, :, 0] = red
            img[:, :, 1] = green
            img[:, :, 2] = blue
            img[:, :, 3] = nir
            img_8bit = self._transfer16bit28bit(img)
            current_save_path = os.path.join(self.save_root_path, "images", f"{current_img}.tif")
            io.imsave(current_save_path, img_8bit)
            print(f"already save {current_save_path}")

    def preprocess(self):
        image_dirs = glob.glob(str(Path(self.root_path) / "*"))
        # print(len(image_dirs))
        tif_dict = self._search_tif(image_dirs)
        self._biome(tif_dict)

    def train_test_split(self):
        """
        Landscape    Scene ID (Level-1T)	 Path	  Row	Acquisition Date	Approximate Cloud Status	Shadows?	Manual CCA
        Barren       LC81570452014213LGN00	 157	  45	8/1/2014	        MidClouds	                no	        61.91%
        Forest       LC80160502014041LGN00	 16	      50	2/10/2014	        MidClouds	                yes	        49.59%
        Forest       LC81750622013304LGN00	 175	  62	10/31/2013	        Clear	                    yes	        7.09%
        Grass/Crops  LC81510262014139LGN00	 151	  26	5/19/2014	        Cloudy	                    no	        100.00%
        Shrubland	 LC81020802014100LGN00	 102	  80	4/10/2014	        Cloudy	                    yes	        76.28%
        Snow/Ice     LC81321192014054LGN00	 132	  119	2/23/2014	        MidClouds	                yes	        20.01%
        Urban        LC81180382014244LGN00	 118	  38	9/1/2014	        Cloudy	                    no	        78.55%
        Water        LC80650182013237LGN00	 65	      18	8/25/2013	        MidClouds	                no	        51.32%
        Wetlands     LC81010142014189LGN00	 101	  14	7/8/2014	        MidClouds	                yes	        64.93%
        """
        test_tif_names = ["LC81570452014213LGN00", "LC80160502014041LGN00", "LC81750622013304LGN00",
                          "LC81510262014139LGN00", "LC81020802014100LGN00", "LC81321192014054LGN00",
                          "LC81180382014244LGN00", "LC80650182013237LGN00", "LC81010142014189LGN00"]
        image_path = os.path.join(self.save_root_path, "images")
        image_test_path = os.path.join(image_path, "test")
        image_train_path = os.path.join(image_path, "train")
        mask_path = os.path.join(self.save_root_path, "masks")
        mask_test_path = os.path.join(mask_path, "test")
        mask_train_path = os.path.join(mask_path, "train")
        false_color_path = os.path.join(self.save_root_path, "images_false_color")
        false_color_test_path = os.path.join(false_color_path, "test")
        false_color_train_path = os.path.join(false_color_path, "train")
        for path in [image_train_path, image_test_path, mask_train_path, mask_test_path, false_color_test_path, false_color_train_path]:
            if not os.path.exists(path):
                os.mkdir(path)
        image_dirs = glob.glob(os.path.join(image_path, "*.tif"))
        for image_dir in tqdm(image_dirs):
            # print(image_dir)
            image_name = os.path.basename(image_dir)
            mask_dir = os.path.join(mask_path, image_name)
            false_color_dir = os.path.join(false_color_path, image_name)
            if image_name.split('.')[0] in test_tif_names:
                image_dst_dir = os.path.join(image_test_path, image_name)
                mask_dst_dir = os.path.join(mask_test_path, image_name)
                false_color_dst_dir = os.path.join(false_color_test_path, image_name)
                shutil.copyfile(image_dir, image_dst_dir)
                shutil.copyfile(mask_dir, mask_dst_dir)
                shutil.copyfile(false_color_dir, false_color_dst_dir)
            else:
                image_dst_dir = os.path.join(image_train_path, image_name)
                mask_dst_dir = os.path.join(mask_train_path, image_name)
                false_color_dst_dir = os.path.join(false_color_train_path, image_name)
                shutil.copyfile(image_dir, image_dst_dir)
                shutil.copyfile(mask_dir, mask_dst_dir)
                shutil.copyfile(false_color_dir, false_color_dst_dir)

    def random_crop(self, crop_size, crop_num):
        masks_root_path = Path(self.save_root_path) / "masks"
        images_root_path = Path(self.save_root_path) / "images"
        false_color_root_path = Path(self.save_root_path) / "images_false_color"

        masks_folder_path = os.path.join(masks_root_path, "train")
        mask_dirs = glob.glob(os.path.join(masks_folder_path, "*.tif"))
        for mask_dir in mask_dirs:
            new_name = 0
            image_name = os.path.basename(mask_dir)
            mask_raw = io.imread(mask_dir)
            image_raw = io.imread(os.path.join(images_root_path, "train", image_name))
            false_color_raw = io.imread(os.path.join(false_color_root_path, "train", image_name))
            height, width = mask_raw.shape[0], mask_raw.shape[1]
            while new_name < crop_num:
                upper_left_x = random.randint(0, height - crop_size)
                upper_left_y = random.randint(0, width - crop_size)
                mask_crop = mask_raw[upper_left_x:upper_left_x + crop_size, upper_left_y:upper_left_y+crop_size]
                if np.sum(mask_crop == 255) > (crop_size*crop_size*0.7):
                    continue
                else:
                    image_crop = image_raw[upper_left_x:upper_left_x + crop_size, upper_left_y:upper_left_y+crop_size]
                    false_color_crop = false_color_raw[upper_left_x:upper_left_x + crop_size, upper_left_y:upper_left_y+crop_size]
                    mask_crop_save = os.path.join(self.save_cut_root_path, "masks", "train", f"{image_name.split('.')[0]}_{new_name}.tif")
                    image_crop_save = os.path.join(self.save_cut_root_path, "images", "train", f"{image_name.split('.')[0]}_{new_name}.tif")
                    false_color_crop_save = os.path.join(self.save_cut_root_path, "images_false_color", "train", f"{image_name.split('.')[0]}_{new_name}.tif")
                    io.imsave(mask_crop_save, mask_crop)
                    io.imsave(image_crop_save, image_crop)
                    io.imsave(false_color_crop_save, false_color_crop)
                    print(f"current img:{image_name}, num:{new_name}")
                    new_name += 1


if __name__ == "__main__":
    # Biome dataset
    biome_root_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_unzip/BC"
    biome_save_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/myBiome"
    biome_cut_save_path = "/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut"
    data_processor = DataProcessor(biome_root_path, biome_save_path, biome_cut_save_path)
    # data_processor.preprocess()
    # data_processor.train_test_split()
    data_processor.random_crop(512, 400)


