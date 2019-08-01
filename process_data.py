"""
Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/process_data.py
"""

import argparse
import os
from glob import glob

import numpy as np

import data_util


def clip_l2_diff(clip, hist_len):
    """
    """
    diff = 0
    for i in range(hist_len):
        frame = clip[:, :, 3 * i : 3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1) : 3 * (i + 2)]
        diff += np.sum(np.square(next_frame - frame))

    return diff


def process_clip(train_dir, frame_h, frame_w, hist_len, movement_threshold=100):
    clip = data_util.get_full_clips(train_dir, hist_len, 1)[0]

    # Randomly crop the clip. With 0.05 probability, take the first crop offered,
    # otherwise, repeat until we have a clip with movement in it.
    take_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([frame_h, frame_w, 3 * (hist_len + 1)])
    for i in range(100):  # cap at 100 trials in case the clip has no movement anywhere
        crop_x = np.random.choice(data_util.FULL_WIDTH - frame_w + 1)
        crop_y = np.random.choice(data_util.FULL_HEIGHT - frame_h + 1)
        cropped_clip = clip[crop_y : crop_y + frame_h, crop_x : crop_x + frame_w, :]

        if take_first or clip_l2_diff(cropped_clip, hist_len) > movement_threshold:
            break

    return cropped_clip


def process_training_data(
    train_dir, num_clips, frame_h, frame_w, hist_len, movement_threshold, clip_dir
):
    num_prev_clips = len(glob(clip_dir + "*"))

    for clip_num in range(num_prev_clips, num_clips + num_prev_clips):
        clip = process_clip(train_dir, frame_h, frame_w, hist_len, movement_threshold)
        np.savez_compressed(os.path.join(clip_dir, str(clip_num)), clip)

        n = clip_num + 1
        if (n) % 100 == 0:
            print("Process {} clips".format(n))


def main():
    parser = argparse.ArgumentParser(description="Script to process video directory.")
    parser.add_argument(
        "-n",
        "--num_clips",
        default=5000000,
        type=int,
        help="Number of clips to output for training.",
    )
    parser.add_argument(
        "-th", "--height", default=32, type=int, help="Frame height of training clips."
    )
    parser.add_argument(
        "-tw", "--width", default=32, type=int, help="Frame width of training clips."
    )
    parser.add_argument(
        "-cl",
        "--hist_len",
        default=4,
        type=int,
        help="Length of training episode clips.",
    )
    parser.add_argument(
        "-t",
        "--move_th",
        default=100,
        type=int,
        help="Movement threshold for l2 diff in considering training clip.",
    )
    parser.add_argument("train_dir", help="Directory containing full training frames.")
    parser.add_argument(
        "clip_dir",
        help=(
            "Output directory for processed training clips."
            "(Make this hidden "
            "so the filesystem doesn't freeze with"
            "so many files. DON'T `ls` THIS DIRECTORY!)"
        ),
    )
    args = parser.parse_args()

    process_training_data(
        args.train_dir,
        args.num_clips,
        args.height,
        args.width,
        args.hist_len,
        args.move_th,
        args.clip_dir,
    )


if __name__ == "__main__":
    main()
