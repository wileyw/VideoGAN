import os
import torch
import torchvision
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from skimage import io, transform
from skimage.transform import resize

import d_net
import g_net

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
        image0 = resize(image0, (32, 32), anti_aliasing=True)
        image0 = np.rollaxis(image0, 2, 0) / 255.

        image_path1 = self.after_image_paths[idx + 1]
        image1 = io.imread(image_path1)
        image1 = resize(image1, (32, 32), anti_aliasing=True)
        image1 = np.rollaxis(image1, 2, 0) / 255.

        return {'image0': image0, 'image1': image1}

def main():
    #dataset = PacmanDataset('Ms_Pacman/Train/')
    dataset = PacmanDataset('Ms_Pacman/Test/')
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=1)

    SCALE_CONV_FSM_D = [[3, 64],
            [3, 64, 128, 128],
            [3, 128, 256, 256],
            [3, 128, 256, 512, 128]]
    SCALE_KERNEL_SIZES_D = [[3],
            [3, 3, 3],
            [5, 5, 5],
            [7, 7, 5, 5]]
    SCALE_FC_LAYER_SIZES_D = [[512, 256, 1],
            [1024, 512, 1],
            [1024, 512, 1],
            [1024, 512, 1]]

    D = d_net.DiscriminatorModel(kernel_sizes_list=SCALE_KERNEL_SIZES_D,
            conv_layer_fms_list=SCALE_CONV_FSM_D,
            scale_fc_layer_sizes_list=SCALE_FC_LAYER_SIZES_D)

    G = g_net.GeneratorDefinitions()

    optimizer = optim.SGD(D.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        for i_batch, sample_batch in enumerate(dataset_loader):
            before_batch = sample_batch['image0'].float()
            after_batch = sample_batch['image1'].float()

            print(before_batch.shape)
            print(after_batch.shape)
            generated_image = G(before_batch)
            print('Size of generated G image')
            print(generated_image.shape)
            exit()

            result = D(generated_image)

            print(result)
            print(result.shape)

            print(i_batch, before_batch.shape, after_batch.shape)
            exit()

if __name__ == '__main__':
    main()
