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

import vanilla_gan
import data_loader


dtype = config.dtype

VIDEO_GAN = True
VANILLA_GAN = not VIDEO_GAN

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

    # D = d_net.DiscriminatorModel(kernel_sizes_list=SCALE_KERNEL_SIZES_D,
    #         conv_layer_fms_list=SCALE_CONV_FSM_D,
    #         scale_fc_layer_sizes_list=SCALE_FC_LAYER_SIZES_D)

    # G = g_net.GeneratorDefinitions()
    # g_optimizer = optim.Adam(G.parameters(), lr=0.001)
    # d_optimizer = optim.Adam(D.parameters(), lr=0.001)

    if VANILLA_GAN:
        vanilla_d_losses = []
        vanilla_g_losses = []

        vanilla_d_net = vanilla_gan.vanilla_gan.Discriminator()
        #vanilla_g_net = vanilla_gan.vanilla_gan.GeneratorSkipConnections()
        vanilla_g_net = vanilla_gan.vanilla_gan.Generator()
        vanilla_d_net.type(dtype)
        vanilla_g_net.type(dtype)
        vanilla_d_optimizer = optim.Adam(vanilla_d_net.parameters(), lr=0.0001)
        vanilla_g_optimizer = optim.Adam(vanilla_g_net.parameters(), lr=0.0001)

        d_num_params = sum(p.numel() for p in vanilla_d_net.parameters())
        g_num_params = sum(p.numel() for p in vanilla_g_net.parameters())
        print('#D parameters:', d_num_params)
        print('#G parameters:', g_num_params)

    if VIDEO_GAN:
        video_d_net = vanilla_gan.video_gan.Discriminator()
        video_g_net = vanilla_gan.video_gan.Generator()
        video_d_net.type(dtype)
        video_g_net.type(dtype)
        video_d_optimizer = optim.Adam(video_d_net.parameters(), lr=0.0001)
        video_g_optimizer = optim.Adam(video_g_net.parameters(), lr=0.0001)


    # Load Pacman dataset
    pacman_dataloader = data_loader.DataLoader('train', 500, 16, 32, 32, 4)

    # Load emojis
    train_dataloader, _ = get_emoji_loader('Windows')

    count = 0
    for i in range(1, 5000):
        for batch in train_dataloader:
            # TESTING: Vanilla Video Gan
            clips_x, clips_y = pacman_dataloader.get_train_batch()
            clips_x = torch.tensor(np.rollaxis(clips_x, 3, 1)).type(dtype)
            clips_y = torch.tensor(np.rollaxis(clips_y, 3, 1)).type(dtype)

            # Before implementing VideoGAN, I implemented a Vanilla GAN from
            # http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
            # Next step is to implement VideoGAN
            real_images, labels = batch
            real_images = Variable(real_images)

            vanilla_d_optimizer.zero_grad()
            vanilla_g_optimizer.zero_grad()

            # TESTING: Vanilla Video Gan
            video_d_optimizer.zero_grad()
            video_g_optimizer.zero_grad()

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100
            sampled_noise = sample_noise(batch_size, noise_size)

            # WGAN loss
            # https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

            if VANILLA_GAN:
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


            if VIDEO_GAN:
                video_images = video_g_net(clips_x)
                # TESTING: Vanilla Video Gan
                video_d_loss_real = (video_d_net(clips_y) - 1).pow(2).mean()
                video_d_loss_fake = (video_d_net(video_images)).pow(2).mean()
                video_d_loss = .5 * (video_d_loss_real + video_d_loss_fake)
                video_d_loss.backward()
                video_d_optimizer.step()

                # batch_size x noise_size x 1 x 1
                batch_size = 16
                noise_size = 100
                sampled_noise = sample_noise(batch_size, noise_size)

                #print('G_Time:', end - start)

                # TESTING: Vanilla Video Gan
                video_images = video_g_net(clips_x)
                video_g_loss_fake = (video_d_net(video_images) - 1).pow(2).mean()
                video_g_loss = video_g_loss_fake
                video_g_loss.backward()
                video_g_optimizer.step()

            if count % 100 == 0:
                print('d_loss_real:', d_loss_real)
                print('d_loss_fake:', d_loss_fake)
                print(generated_images.shape)
                print('g_loss:', g_loss)

                print('Mean')
                print(torch.mean(generated_images))

                print(generated_images.shape)
                save_samples(real_images, count, "real")
                save_samples(generated_images, count, "fake")

                save_samples(clips_y, count, "video_real")
                save_samples(video_images, count, "video_fake")
            count += 1

            if VANILLA_GAN:
                # Record loss values
                vanilla_d_losses.append(d_loss)
                vanilla_g_losses.append(g_loss)

    matplotlib.pyplot.plot(vanilla_d_losses)
    matplotlib.pyplot.plot(vanilla_g_losses)
    plt.savefig('plot.jpg')
    plt.show()


if __name__ == '__main__':
    main()
