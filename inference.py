import argparse

import cv2
import torch
import numpy as np

import data_util
from vanilla_gan.video_gan import VideoGANGenerator


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


def save_to_video(frames, video_filename):
    video = cv2.VideoWriter(video_filename)
    for frame in frames:
        video.write(frame)
    video.release()


def main():
    parser = argparse.ArgumentParser(
        description="""
            Script to run inference on a random input seeding
            of videoGAN
        """)
    parser.add_argument(
        'video_filename',
        help='output filename of video')
    parser.add_argument(
        '-i', '--input_dir',
        default='Ms_Pacman/Test',
        help='input dir of training to take random images from')


    args = parser.parse_args()

    # Load generator.
    generator = VideoGANGenerator()
    print("blablabla")
    generator.load_state_dict(torch.load(MODEL_FILEPATH))
    print("BLABLABLA")
    generator.eval()

    # Load input seed.
    frames = data_util.get_full_clips()[:HIST_LEN]
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

    save_to_video(frames, args.video_filename)

if __name__ == '__main__':
    main()
