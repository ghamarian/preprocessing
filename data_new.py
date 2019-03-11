from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder

ROOT_DIR = "./data"


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


class DatasetFactory:
    DatasetMap = {'file': DatasetFromFolder}

    def __init__(self, upscale_factor, crop_size, which_type='file'):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.datasetFactory = self.DatasetMap[which_type]

    def calculate_valid_crop_size(self):
        return self.crop_size - (self.crop_size % self.upscale_factor)

    def input_transform(self, crop_size):
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // self.upscale_factor),
            ToTensor(),
        ])

    def target_transform(self, crop_size):
        return Compose([
            CenterCrop(crop_size),
            ToTensor(),
        ])

    def get_training_set(self):
        # root_dir = download_bsd300()
        root_dir = ROOT_DIR
        train_dir = join(root_dir, "train")
        crop_size = self.calculate_valid_crop_size()

        return self.datasetFactory(train_dir,
                                   input_transform=self.input_transform(crop_size),
                                   target_transform=self.target_transform(crop_size))

    def get_test_set(self):
        root_dir = download_bsd300()
        test_dir = join(root_dir, "test")
        crop_size = self.calculate_valid_crop_size()

        return self.datasetFactory(test_dir,
                                   input_transform=self.input_transform(crop_size),
                                   target_transform=self.target_transform(crop_size))
