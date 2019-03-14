# %%
from pylab import *
from skimage.morphology import watershed
import scipy.ndimage as ndimage
from PIL import Image, ImagePalette
import argparse
import pdb

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

import tifffile as tiff
import cv2
import random
from pathlib import Path
import os
from data_new import DatasetFactory
from losses import bce_dice_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torch.nn import functional as F
from models.ternausnet2 import TernausNetV2

# %%
random.seed(42)
NUCLEI_PALETTE = ImagePalette.random()
random.seed()

# %%
rcParams['figure.figsize'] = 15, 15

# %%
from models.ternausnet2 import TernausNetV2

# %%
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")
IMAGE_DIR = "/projects/asm/data/m2l/crops"


# %%
def get_model():
    model = TernausNetV2(opt.batchSize, num_classes=2)

    if torch.cuda.is_available():
        model.cuda()
    return model


# %%
def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


# %%
def label_watershed(before, after, component_size=20):
    markers = ndimage.label(after)[0]

    labels = watershed(-before, markers, mask=before, connectivity=8)
    unique, counts = np.unique(labels, return_counts=True)

    for (k, v) in dict(zip(unique, counts)).items():
        if v < component_size:
            labels[labels == k] = 0
    return labels


# %%
def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        batch = [b.to(device) for b in batch]
        input_img, target1, target2 = batch

        optimizer.zero_grad()

        y_pred = torch.sigmoid(model(input_img))
        loss = criterion(target1, target2, y_pred)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): Loss: {loss.item():.4f}")

    print(f"===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / len(training_data_loader):.4f}")


def dice_score(target1, target2, prediction):
    y_true_f = target1.view(1, -1).squeeze()
    prediction = (prediction > 0.5).int()
    y_pred_f = prediction.contiguous().view(1, -1).squeeze()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f))


def runtest():
    avg_psnr = 0
    dice = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            batch = [b.to(device) for b in batch]
            input_img, target1, target2 = batch

            prediction = torch.sigmoid(model(input_img))
            loss = criterion(target1, target2, prediction)
            dice += dice_score(target1, target2, prediction)
            avg_psnr += loss.item()
    print(f"===> Avg. loss: {avg_psnr / len(testing_data_loader):.4f} dB")
    print(f"===> Avg. dice: {dice / len(testing_data_loader):.4f} dB")


def checkpoint(epoch):
    model_out_path = f"model_epoch_{epoch}.pth"
    torch.save(model, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


if __name__ == '__main__':

    print('===> Creating the model')
    model = get_model()
    criterion = bce_dice_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    print('===> Loading datasets')
    data_gen = DatasetFactory(opt.upscale_factor, 256, IMAGE_DIR)
    train_set = data_gen.get_training_set()
    test_set = data_gen.get_test_set()

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        runtest()
        checkpoint(epoch)
