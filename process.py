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
import g_net
import config
import time

dtype = config.dtype

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

    if not os.path.exists('output'):
        os.makedirs('output')

    #img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output/test{:05d}.jpg'.format(i), img_data)

def save_samples(generated_images, iteration, prefix):
    import scipy
    generated_images = generated_images.data.cpu().numpy()

    num_images, channels, cell_h, cell_w = generated_images.shape
    ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=generated_images.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = generated_images[i*ncols+j].transpose(1, 2, 0)
    grid = result

    if not os.path.exists('output'):
        os.makedirs('output')
    scipy.misc.imsave('output/{}_{:05d}.jpg'.format(prefix, iteration), grid)


def sample_noise(batch_size, dim):
    result = torch.rand(batch_size, dim) * 2 - 1
    result = Variable(result).unsqueeze(2).unsqueeze(3)

    return result

def get_emoji_loader(emoji_type):
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader

    num_workers = 1
    batch_size = 16
    image_size = 32

    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_path = os.path.join('./emojis', emoji_type)
    test_path = os.path.join('./emojis', 'Test_{}'.format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dloader, test_dloader

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

    vanilla_d_losses = []
    vanilla_g_losses = []

    import vanilla_gan.vanilla_gan
    vanilla_d_net = vanilla_gan.vanilla_gan.Discriminator()
    vanilla_g_net = vanilla_gan.vanilla_gan.GeneratorSkipConnections()
    vanilla_d_net.type(dtype)
    vanilla_g_net.type(dtype)
    #vanilla_d_optimizer = optim.Adam(vanilla_d_net.parameters(), lr=0.0003)
    #vanilla_g_optimizer = optim.Adam(vanilla_g_net.parameters(), lr=0.0003)
    vanilla_d_optimizer = optim.Adam(vanilla_d_net.parameters(), lr=0.0001)
    vanilla_g_optimizer = optim.Adam(vanilla_g_net.parameters(), lr=0.0001)

    d_num_params = sum(p.numel() for p in vanilla_d_net.parameters())
    g_num_params = sum(p.numel() for p in vanilla_g_net.parameters())
    print('#D parameters:', d_num_params)
    print('#G parameters:', g_num_params)

    # Load emojis
    train_dataloader, _ = get_emoji_loader('Windows')

    save_dummy_data(dog_data, 0)
    count = 0
    for i in range(1, 5000):
        for batch in train_dataloader:
            # Before implementing VideoGAN, I implemented a Vanilla GAN from
            # http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
            # Next step is to implement VideoGAN
            real_images, labels = batch
            real_images = Variable(real_images)

            vanilla_d_optimizer.zero_grad()
            vanilla_g_optimizer.zero_grad()

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100
            sampled_noise = sample_noise(batch_size, noise_size)

            # WGAN loss
            # https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

            # Step 1. Make one discriminator step
            start = time.time()
            generated_images = vanilla_g_net(sampled_noise)
            if config.use_wgan_loss:
                d_loss_real = (vanilla_d_net(real_images) * 1.0).mean()
                d_loss_fake = (vanilla_d_net(generated_images) * -1.0).mean()
            else:
                d_loss_real = (vanilla_d_net(real_images) - 1).pow(2).mean()
                d_loss_fake = (vanilla_d_net(generated_images)).pow(2).mean()
            d_loss = .5 * (d_loss_fake + d_loss_real)
            d_loss.backward()
            vanilla_d_optimizer.step()
            end = time.time()
            #print('D_Time:', end - start)

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100
            sampled_noise = sample_noise(batch_size, noise_size)

            # Step 2. Make one generator step
            start = time.time()
            generated_images = vanilla_g_net(sampled_noise)
            if config.use_wgan_loss:
                g_loss_fake = (vanilla_d_net(generated_images) * 1.0).mean()
            else:
                g_loss_fake = (vanilla_d_net(generated_images) - 1).pow(2).mean()
            g_loss = g_loss_fake
            g_loss.backward()
            vanilla_g_optimizer.step()
            end = time.time()
            #print('G_Time:', end - start)

            if count % 100 == 0:
                print('d_loss_real:', d_loss_real)
                print('d_loss_fake:', d_loss_fake)
                print(generated_images.shape)
                print(dog_data.shape)
                print('g_loss:', g_loss)

                print('Mean')
                print(torch.mean(generated_images))

                print(generated_images.shape)
                save_samples(real_images, count, "real")
                save_samples(generated_images, count, "fake")
            count += 1

            # Record loss values
            vanilla_d_losses.append(d_loss)
            vanilla_g_losses.append(g_loss)

    matplotlib.pyplot.plot(vanilla_d_losses)
    matplotlib.pyplot.plot(vanilla_g_losses)
    plt.savefig('plot.jpg')
    plt.show()


if __name__ == '__main__':
    main()
