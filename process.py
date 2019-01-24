import os
import torch
import torchvision
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import cv2
import matplotlib
import matplotlib.pyplot as plt

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
        HIST_LEN = 1
        for image_dir in image_dirs:
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            for i in range(len(image_paths) - HIST_LEN):
                self.before_image_paths.append(image_paths[i:i + HIST_LEN])
                self.after_image_paths.append(image_paths[i + HIST_LEN])

    def __len__(self):
        return len(self.before_image_paths)

    def __getitem__(self, idx):
        history_images = []
        for image_path0 in self.before_image_paths[idx]:
            image0 = io.imread(image_path0)
            image0 = resize(image0, (32, 32), anti_aliasing=True)
            image0 = np.rollaxis(image0, 2, 0) / 255.
            history_images.append(torch.tensor(image0))

        stacked_history = torch.cat(history_images, 0)

        image_path1 = self.after_image_paths[idx + 1]
        image1 = io.imread(image_path1)
        image1 = resize(image1, (32, 32), anti_aliasing=True)
        image1 = np.rollaxis(image1, 2, 0) / 255.

        return {'image0': stacked_history, 'image1': image1}

def load_dummy_data():
    dog_img = cv2.imread('dog.jpg')
    dog_img = cv2.resize(dog_img, (32, 32))

    dog_data = np.rollaxis(dog_img, 2, 0) / 255.
    dog_data = np.expand_dims(dog_data, axis=0)

    dog_data = torch.from_numpy(dog_data).float()

    return dog_data

def display_dummy_data(dummy_data):
    dummy_data = dummy_data.numpy()

    img_data = np.squeeze(dummy_data)
    img_data = np.rollaxis(img_data, 0, 3) * 255.
    img_data = img_data.astype(np.uint8)

    cv2.imshow('img', img_data)
    cv2.waitKey(0)

def save_dummy_data(dummy_data, i):
    dummy_data = dummy_data.detach().numpy()

    img_data = np.squeeze(dummy_data)
    img_data = np.rollaxis(img_data, 0, 3) * 255.
    img_data = img_data.astype(np.uint8)

    cv2.imwrite('test{}.jpg'.format(i), img_data)

def main():
    #dataset = PacmanDataset('Ms_Pacman/Train/')
    dataset = PacmanDataset('Ms_Pacman/Test/')
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

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
    import dummy_net
    Dummy = dummy_net.Dummy()

    # Load dummy data
    dog_data = load_dummy_data()

    # Display dummy data
    #display_dummy_data(dog_data)

    print('Dummy Parameters', list(Dummy.parameters()))
    dummy_optimizer = optim.Adam(Dummy.parameters(), lr=0.1)
    g_optimizer = optim.Adam(G.parameters(), lr=0.001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.001)

    dummy_values = []
    for epoch in range(1000):
        for i_batch, sample_batch in enumerate(dataset_loader):
            before_batch = sample_batch['image0'].float()
            after_batch = sample_batch['image1'].float()

            # Clear gradients before calling loss.backward() and optimizer.step()
            dummy_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Define the Dummy loss
            dummy_loss = Dummy(-5).pow(2)
            dummy_param = list(Dummy.parameters())[0].tolist()[0]
            print('Dummy Loss:', dummy_loss)
            print('Dummy Solution:', dummy_param)

            # Define the Generator Loss function
            # TODO: Init the Generator weights to something reasonable
            generated_image = G(before_batch)
            g_loss = (generated_image - dog_data).pow(2)
            g_loss = g_loss.sum(1).sum(1).sum(1)
            print(g_loss.shape)
            print(g_loss.detach().numpy())
            print('Generator Loss:', g_loss)

            # Define a Vanilla GAN
            """
            d_on_real = (D(dog_data) - 1).pow(2) + D(G(before_batch)).pow(2)
            d_loss = d_on_real

            generated_image = G(before_batch)
            g_on_real = (D(generated_image) - 1).pow(2)
            g_loss = g_on_real
            print('d_on_real:', d_loss)
            print('g_on_real:', g_loss)
            """

            # Dummy back prop and optimizer step
            dummy_loss.backward()
            dummy_optimizer.step()

            # Generator back prop and optimizer step
            g_loss.backward()
            g_optimizer.step()

            # Discriminator loss
            """
            d_loss.backward()
            d_optimizer.step()
            """

            # Save values to plot
            dummy_values.append(dummy_param)

            save_dummy_data(generated_image, epoch)

            break

    print(dummy_values)
    matplotlib.pyplot.plot(dummy_values)
    plt.show()


if __name__ == '__main__':
    main()
