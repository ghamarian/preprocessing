import torch.utils.data as data

from os import listdir
from os.path import join, splitext
import numpy as np
import cv2
from PIL import Image


IMAGE_SIZE = 704
IMAGE_DIM = (IMAGE_SIZE, IMAGE_SIZE)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".jpeg"])


def is_not_mask(filename):
    return 'mask' not in filename


def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def load_img(file_name_rgb):
    im = Image.open(str(file_name_rgb))  # Can be many different formats.
    # rgb = im.load()
    # rgb = minmax(rgb)
    # img, pads = pad(rgb)
    # input_img = torch.unsqueeze(img_transform(img / (2 ** 8 - 1)).cuda(), dim=0)
    # return np.concatenate([rgb, tf], axis=2) * (2 ** 8 - 1)
    return im.resize(IMAGE_DIM)



def minmax(img):
    out = np.zeros_like(img).astype(np.float32)
    if img.sum() == 0:
        return bands

    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    return out.astype(np.float32)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x) and is_not_mask(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input, target1, target2 = self.load_files(index)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target1 = self.target_transform(target1)
        if self.target_transform:
            target2 = self.target_transform(target2)

        return input, target1, target2

    def load_files(self, index):
        input_filename = self.image_filenames[index]
        basename, ext = splitext(input_filename)
        target1_name = f'{basename}_mask{ext}'
        target2_name = f'{basename}_mask{ext}'

        input = load_img(input_filename)
        target1 = load_img(target1_name)
        target2 = load_img(target2_name)
        return input, target1, target2

    def __len__(self):
        return len(self.image_filenames)
