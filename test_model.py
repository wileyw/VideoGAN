import os
import torch
import torchvision
import glob
import numpy as np
import torch.optim as optim
import cv2
import matplotlib
import matplotlib.pyplot as plt

from skimage import io, transform
from skimage.transform import resize
from torch.autograd import Variable

import d_net
import config
import time

import vanilla_gan
import vanilla_gan.vanilla_gan
import vanilla_gan.video_gan
import data_loader
import loss_funs



MODEL_FILEPATH = 'generator_net.pth.tmp'
NUM_RECURSIONS = 2# 64
HIST_LEN = 4
CROP_HEIGHT = 32
CROP_WIDTH = 32
DTYPE = config.dtype

def test():
    count = 0

    video_g_net = VideoGANGenerator()
    video_g_net.load_state_dict(torch.load(MODEL_FILEPATH))
    video_g_net.eval()

    clips_x, clips_y = pacman_dataloader.get_train_batch()
    clips_x = torch.tensor(np.rollaxis(clips_x, 3, 1)).to(device)
    clips_y = torch.tensor(np.rollaxis(clips_y, 3, 1)).to(device)

    # batch_size x noise_size x 1 x 1
    batch_size = 16
    noise_size = 100

    video_images = video_g_net(clips_x)

    save_samples(video_images, count, "test_model")


def main():
    test_model()


if __name__ == '__main__':
    main()
