import argparse

import cv2
import torch
import numpy as np

import data_util
import g_net


MODEL_FILEPATH = 'generator_net.pth'
NUM_RECURSIONS = 64
HIST_LEN = 4
CROP_HEIGHT = 32
CROP_WIDTH = 32


def crop_batch(images, fw, fh, cw, ch):
    batch = []
    for i in range(fw/cw):
        for j in range(fh/ch):
            batch.append([img[i * cw: (i + 1) * cw,
                              j * ch: (j + 1) * ch]])
    return batch


def reconstruct_frame(result, fw, fh, cw, ch):
    out = np.empty([fw, fh, 3])
    for i in range(fw/cw):
        for j in range(fh/ch):
            out[i * cw: (i + 1) * cw,
                j * ch: (j + 1) * ch,
                :] = result[i * (fh/ch) + j]


def main():
    # Load generator.
    generator = g_net.GeneratorDefinitions()
    generator.load_state_dict(torch.load(MODEL_FILEPATH))
    generator.eval()

    # Load input seed.
    # TODO: implement frame load seed
    frames = [None, None, None]
    frame_w, frame_h = frames[0].shape

    # Set initial frames.
    input_frames = data_util.normalize_frames(frames[i, i + HIST_LEN])
    input_batched = crop_batch(input_frames)

    # Run inference for length of recursions.
    for i in range(NUM_RECURSIONS):

        # Run inference and reconstruct frame.
        result = generator(input_batched)
        result_reconst = reconstruct_frame(result)
        result_denorm = data_util.denormalize_frames(result_reconst)
        frames.append(result_denorm)

        # Pop first frame and add result to input.
        input_batched.pop(0)
        input_batched.append(result)

    # TODO: save frames into video.


if __name__ == '__main__':
    main()
