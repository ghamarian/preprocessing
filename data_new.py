import torch

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize, Lambda

from dataset import DatasetFromFolder


class DatasetFactory:
    DatasetMap = {'file': DatasetFromFolder}

    def __init__(self, upscale_factor, crop_size, root_dir, which_type='file'):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.datasetFactory = self.DatasetMap[which_type]
        self.root_dir = root_dir

    def calculate_valid_crop_size(self):
        return self.crop_size - (self.crop_size % self.upscale_factor)

    def input_transform(self, crop_size):
        return Compose([
            ToTensor(),
        ])

    def target_transform(self, crop_size):
        return Compose([
            ToTensor(),
        ])

    def get_training_set(self):
        # root_dir = download_bsd300()
        # train_dir = join(self.root_dir, "train")
        train_dir = self.root_dir
        crop_size = self.calculate_valid_crop_size()

        return self.datasetFactory(train_dir,
                                   input_transform=self.input_transform(crop_size),
                                   target_transform=self.target_transform(crop_size))

    def get_validation_set(self):
        # root_dir = download_bsd300()
        # validation_dir = join(self.root_dir, "valid")
        validation_dir = self.root_dir
        crop_size = self.calculate_valid_crop_size()

        return self.datasetFactory(validation_dir,
                                   input_transform=self.input_transform(crop_size),
                                   target_transform=self.target_transform(crop_size))

    def get_test_set(self):
        # root_dir = download_bsd300()
        # test_dir = join(self.root_dir, "test")
        test_dir = self.root_dir
        crop_size = self.calculate_valid_crop_size()

        return self.datasetFactory(test_dir,
                                   input_transform=self.input_transform(crop_size),
                                   target_transform=self.target_transform(crop_size))