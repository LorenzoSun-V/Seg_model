import cv2
import random
import numpy as np


def soft_resize( src, h, w ):
    tmp = cv2.resize( src, (2*w, 2*h) ).astype( np.float32 )
    A = tmp[0:2*h:2, 0:2*w:2, :]
    B = tmp[0:2*h:2, 1:2*w:2, :]
    C = tmp[1:2*h:2, 0:2*w:2, :]
    D = tmp[1:2*h:2, 1:2*w:2, :]
    return ( 0.25 / 255 ) * ( A + B + C + D )


class Brightness(object):
    def __init__(self, prob=0.5, factor=10):
        self.factor = factor
        self.prob = prob

    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.prob:
            return img, mask

        if random.uniform(0, 1) > self.prob:
            img = img.astype(np.float32)
            shift = random.randint(-self.factor, self.factor)
            img[:, :, :] += shift
            img = np.around(img)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img, mask


class ScaleCrop(object):
    """
    With a probability, first increase img size to (1 + 1/8), and then perform random crop.
    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, scale, p=0.5):
        self.height = height
        self.width = width
        self.scale = scale
        self.p = p

    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.p:
            return cv2.resize(img, (self.width, self.height), cv2.INTER_LINEAR), \
                   cv2.resize(mask, (self.width, self.height), cv2.INTER_NEAREST),
        new_width, new_height = int(round(self.width * self.scale)), int(round(self.height * self.scale))
        resized_img = cv2.resize(img, (new_width, new_height), cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (new_width, new_height), cv2.INTER_NEAREST)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img[y1: y1 + self.height, x1: x1 + self.width]
        croped_mask = resized_mask[y1: y1 + self.height, x1: x1 + self.width]
        return croped_img, croped_mask


class Blur(object):
    def __init__(self, prob=0.5, mode='random', kernel_size=3, sigma=1):
        self.prob = prob
        self.mode = mode
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, mask):
        if random.uniform(0, 1) < self.prob:
            return img, mask

        if self.mode == 'random':
            self.mode = random.choice(['normalized', 'gaussian', 'median'])

        if self.mode == 'normalized':
            result = cv2.blur(img, (self.kernel_size, self.kernel_size))
        elif self.mode == 'gaussian':
            result = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigmaX=self.sigma, sigmaY=self.sigma)
        elif self.mode == 'median':
            result = cv2.medianBlur(img, self.kernel_size)
        else:
            print('Blur mode is not supported: %s.' % self.mode)
            result = img
        return result, mask


class Rotation(object):
    def __init__(self, prob=0.5, degree=10, mode='crop'):
        self.prob = prob
        self.degree = random.randint(-1*degree, degree)
        self.mode = mode

    def __call__(self, img, mask):
        if random.uniform(0, 1) > self.prob:
            h, w = img.shape[:2]
            center_x, center_y = w / 2, h / 2
            M = cv2.getRotationMatrix2D((center_x, center_y), self.degree, scale=1)

            if self.mode == 'crop':  # keep original size
                new_w, new_h = w, h
            else:  # keep full img
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w / 2) - center_x
                M[1, 2] += (new_h / 2) - center_y

            result_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))
            result_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=(255, 255, 255))
            return result_img, result_mask
        else:
            return img, mask


class Flip(object):
    def __init__(self, prob=0.5, mode='h'):
        self.prob = prob
        self.mode = mode

    def __call__(self, img, mask):
        if random.uniform(0, 1) > self.prob:
            if self.mode == 'h':
                return cv2.flip(img, 1), cv2.flip(mask, 1)
            elif self.mode == 'v':
                return cv2.flip(img, 0), cv2.flip(mask, 0)
            else:
                print('Unsupported mode: %s.' % self.mode)
                return img, mask
        return img, mask


class Resize(object):
    def __init__(self, size_in_pixel=None, size_in_scale=None):
        """
        :param size_in_pixel: tuple (width, height)
        :param size_in_scale: tuple (width_scale, height_scale)
        :return:
        """
        self.size_in_pixel = size_in_pixel
        self.size_in_scale = size_in_scale

    def __call__(self, img, mask):
        if self.size_in_pixel is not None:
            return cv2.resize(img, self.size_in_pixel, interpolation=cv2.INTER_LINEAR), \
            cv2.resize(mask, self.size_in_pixel, interpolation=cv2.INTER_NEAREST)
        elif self.size_in_scale is not None:
            return cv2.resize(img, (0, 0), fx=self.size_in_scale[0], fy=self.size_in_scale[1], interpolation=cv2.INTER_LINEAR), \
                   cv2.resize(mask, (0, 0), fx=self.size_in_scale[0], fy=self.size_in_scale[1], interpolation=cv2.INTER_NEAREST)
        else:
            print('size_in_pixel and size_in_scale are both None.')
            return img, mask


if __name__ == "__main__":
    rotation = Rotation(prob=0, degree=50)
    colorjitter = Brightness(prob=0, factor=50)
    # random_dark = RandomDark()
    blur = Blur(prob=0, mode='random')  # ['normalized', 'gaussian', 'median']
    flip = Flip(prob=0, mode='h')
    scale_crop = ScaleCrop(height=576, width=1024, scale=1.125, p=0)
    img = cv2.imread("/mnt2/sjh/seg_data/myCityspaces/images/train/aachen_000000_000019.png")
    mask = cv2.imread("/mnt2/sjh/seg_data/myCityspaces/masks/train/aachen_000000_000019.png", -1)
    # img, mask = rotation(img=img, mask=mask)
    # img, mask = colorjitter(img=img, mask=mask)
    # img, mask = blur(img=img, mask=mask)
    # img, mask = flip(img=img, mask=mask)
    img, mask = scale_crop(img=img, mask=mask)
    from utils.cityspaces_vis import addmask2img
    new_img = addmask2img(img, mask)
    cv2.imwrite("/mnt/shy/sjh/test_aug/a.png", new_img)
    # cv2.imwrite("/mnt/shy/sjh/test_aug/aug/DJI_0013_000135--000155_0001_1631513816.jpg", result_img)
    # cv2.imwrite("/mnt/shy/sjh/test_aug/aug/DJI_0013_000135--000155_0001_1631513816.png", result_mask)
