from PIL import Image
import numpy as np
from skimage import io


a = Image.open("/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut/images/train/LC80010112014080LGN00_13.tif")
a_n = np.unique(a)
b = io.imread("/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Biome_cut/images/train/LC80010112014080LGN00_13.tif")
b_n = np.unique(b)
if (np.array(a) == b).any():
    print("yes")
print(np.array(a).shape)
print(b.shape)