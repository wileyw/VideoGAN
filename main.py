import os
import torch
import torchvision
import glob
import matplotlib.pyplot as plt
import numpy as np

from skimage import io, transform

class PacmanDataset(torch.utils.data.Dataset):
    def __init__(self, videos_dir):
        self.videos_dir = videos_dir
        image_dirs = glob.glob(os.path.join(self.videos_dir, '*'))

        self.before_image_paths = []
        self.after_image_paths = []
        for image_dir in image_dirs:
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            for i in range(len(image_paths) - 1):
                self.before_image_paths.append(image_paths[i])
                self.after_image_paths.append(image_paths[i + 1])

    def __len__(self):
        return len(self.before_image_paths)

    def __getitem__(self, idx):
        image_path0 = self.before_image_paths[idx]
        image0 = io.imread(image_path0)

        image_path1 = self.after_image_paths[idx + 1]
        image1 = io.imread(image_path1)

        return {'image0': image0, 'image1': image1}

def main():
    dataset = PacmanDataset('Ms_Pacman/Train/')
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    for i_batch, sample_batch in enumerate(dataset_loader):
        before_batch = sample_batch['image0']
        after_batch = sample_batch['image1']
        print(i_batch, before_batch.shape, after_batch.shape)

if __name__ == '__main__':
    main()
