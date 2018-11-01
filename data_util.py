"""
Adapted from https://github.com/dyelax/Adversarial_Video_Generation/blob/master/Code/utils.py
"""
import os
from glob import glob

import numpy as np
from scipy.ndimage import imread


FULL_HEIGHT = 210
FULL_WIDTH = 160
TRAIN_HEIGHT = TRAIN_WIDTH = 32


def normalize_frames(frames):
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames


def denormalize_frames(frames):
    new_frames = frames + 1
    new_frames *= (255 / 2)
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def get_full_clips(data_dir, hist_len, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.
    """
    clips = np.empty([num_clips,
                      FULL_HEIGHT,
                      FULL_WIDTH,
                      (3 * (hist_len + num_rec_out))])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (hist_len + num_rec_out - 1))
        clip_frame_paths = ep_frame_paths[start_index:start_index + (hist_len + num_rec_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            norm_frame = normalize_frames(frame)

            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = norm_frame

    return clips


def get_train_batch(train_dir, num_clips, batch_size, frame_w, frame_h, hist_len):
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
    """
    clips = np.empty([batch_size, frame_w, frame_h, (3 * (hist_len + 1))],
                     dtype=np.float32)
    for i in xrange(batch_size):
        path = train_dir + str(np.random.choice(num_clips)) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip

    return clips


def get_test_batch(test_dir, test_batch_size, num_rec_out=1):
    """
    Gets a clip from the test dataset.

    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape:
             [test_batch_size, c.TEST_HEIGHT, c.TEST_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    return get_full_clips(test_dir, test_batch_size, num_rec_out=num_rec_out)
