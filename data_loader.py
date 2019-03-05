import os
import numpy as np


class DataLoader:
    def __init__(self,
                 train_dir_clips,
                 num_clips,
                 batch_size,
                 train_height,
                 train_width,
                 hist_len):
        self.batch_size = batch_size
        self.train_height = train_height
        self.train_width = train_width
        self.hist_len = hist_len
        self.num_clips = num_clips
        self.train_dir_clips = train_dir_clips

    def get_train_batch(self):
        """
        Loads batch_size clips from the database of preprocessed training clips.

        @return: An array of shape
                [batch_size, train_height, train_width, (3 * (hist_len + 1))].
        """
        clips = np.empty([self.batch_size,
                          self.train_height,
                          self.train_width,
                          (3 * (self.hist_len + 1))],
                         dtype=np.float32)
        for i in range(self.batch_size):
            path = os.path.join(
                self.train_dir_clips,
                str(np.random.choice(self.num_clips)) +
                '.npz')
            clip = np.load(path)['arr_0']

            clips[i] = clip

        return clips
