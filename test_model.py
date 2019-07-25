import config
import numpy as np
import os
import scipy
import torch

import data_loader

from video_gan import VideoGANGenerator


MODEL_FILEPATH = "generator_net.pth.tmp"
NUM_RECURSIONS = 2  # 64
HIST_LEN = 4
CROP_HEIGHT = 32
CROP_WIDTH = 32
DTYPE = config.dtype


def save_samples(generated_images, iteration, prefix):

    generated_images = generated_images.data.cpu().numpy()

    num_images, channels, cell_h, cell_w = generated_images.shape
    ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels), dtype=generated_images.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w, :
            ] = generated_images[i * ncols + j].transpose(1, 2, 0)
    grid = result

    if not os.path.exists("output"):
        os.makedirs("output")
    scipy.misc.imsave("output/{}_{:05d}.jpg".format(prefix, iteration), grid)


def test_model():
    count = 0

    video_g_net = VideoGANGenerator()
    video_g_net.load_state_dict(torch.load(MODEL_FILEPATH))
    video_g_net.eval()

    max_size = len(os.listdir("train"))
    pacman_dataloader = data_loader.DataLoader(
        "train", min(max_size, 500000), 16, 32, 32, 4
    )
    clips_x, clips_y = pacman_dataloader.get_train_batch()
    clips_x = torch.tensor(np.rollaxis(clips_x, 3, 1))
    clips_y = torch.tensor(np.rollaxis(clips_y, 3, 1))

    video_images = video_g_net(clips_x)

    save_samples(video_images, count, "test_model")


def main():
    test_model()


if __name__ == "__main__":
    main()
