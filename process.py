import os
import torch
import numpy as np
import torch.optim as optim

import config
import data_loader
import d_net
import loss_funs
import video_gan

dtype = config.dtype


def save_samples(generated_images, iteration, prefix):
    import scipy

    generated_images = generated_images.data.cpu().numpy()

    num_images, channels, cell_h, cell_w = generated_images.shape
    ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels), dtype=generated_images.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w, :
            ] = generated_images[i * ncols + j].transpose(1, 2, 0)
    grid = result

    if not os.path.exists("output"):
        os.makedirs("output")
    scipy.misc.imsave("output/{}_{:05d}.jpg".format(prefix, iteration), grid)


def get_emoji_loader(emoji_type):
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader

    num_workers = 1
    batch_size = 16
    image_size = 32

    transform = transforms.Compose(
        [
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_path = os.path.join("./emojis", emoji_type)
    test_path = os.path.join("./emojis", "Test_{}".format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dloader, test_dloader


def main():
    SCALE_CONV_FSM_D = [
        [3, 64],
        [3, 64, 128, 128],
        [3, 128, 256, 256],
        [3, 128, 256, 512, 128],
    ]
    SCALE_KERNEL_SIZES_D = [[3], [3, 3, 3], [5, 5, 5], [7, 7, 5, 5]]
    SCALE_FC_LAYER_SIZES_D = [
        [512, 256, 1],
        [1024, 512, 1],
        [1024, 512, 1],
        [1024, 512, 1],
    ]

    loss_fp = open("losses.csv", "w")

    video_d_net = d_net.DiscriminatorModel(
        kernel_sizes_list=SCALE_KERNEL_SIZES_D,
        conv_layer_fms_list=SCALE_CONV_FSM_D,
        scale_fc_layer_sizes_list=SCALE_FC_LAYER_SIZES_D,
    )
    video_d_net.type(dtype)

    video_g_net = video_gan.VideoGANGenerator()
    video_g_net.type(dtype)

    video_d_optimizer = optim.Adam(video_d_net.parameters(), lr=0.0001)
    video_g_optimizer = optim.Adam(video_g_net.parameters(), lr=0.0001)

    # Load Pacman dataset
    max_size = len(os.listdir("train"))
    pacman_dataloader = data_loader.DataLoader(
        "train", min(max_size, 500000), 16, 32, 32, 4
    )

    # Load emojis
    train_dataloader, _ = get_emoji_loader("Windows")

    count = 0
    for i in range(1, 5000):
        for batch in train_dataloader:
            clips_x, clips_y = pacman_dataloader.get_train_batch()
            clips_x = torch.tensor(np.rollaxis(clips_x, 3, 1)).type(dtype)
            clips_y = torch.tensor(np.rollaxis(clips_y, 3, 1)).type(dtype)

            video_d_optimizer.zero_grad()
            video_g_optimizer.zero_grad()

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100

            # WGAN loss
            # https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

            video_images = video_g_net(clips_x)
            # TESTING: Vanilla Video Gan
            video_d_loss_real = (video_d_net(clips_y) - 1).pow(2).mean()
            video_d_loss_fake = (video_d_net(video_images)).pow(2).mean()

            # Fake batch
            labels = torch.zeros(batch_size, 4).t().unsqueeze(2).type(dtype)
            video_d_loss_fake = loss_funs.adv_loss(
                video_d_net(video_images), labels
            )  # TODO: Validate if it's right.
            video_d_optimizer.zero_grad()
            video_d_loss_fake.backward()
            video_d_optimizer.step()

            # Real batch
            labels = torch.ones(batch_size, 4).t().unsqueeze(2).type(dtype)
            video_d_loss_real = loss_funs.adv_loss(
                video_d_net(clips_y), labels
            )  # TODO: Validate if it's right.
            video_d_optimizer.zero_grad()
            video_d_loss_real.backward()
            video_d_optimizer.step()

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100

            # print('G_Time:', end - start)

            # TESTING: Vanilla Video Gan
            video_images = video_g_net(clips_x)
            video_g_loss_fake = (video_d_net(video_images) - 1).pow(2).mean()
            d_preds = video_d_net(video_images).type(
                dtype
            )  # TODO: Make sure this is working.
            gt_frames = clips_y.type(
                dtype
            )  # TODO: make clips_y at different scales.
            gen_frames = video_images.type(
                dtype
            )  # TODO: make the generated frames multi scale.
            video_g_loss = loss_funs.combined_loss(gen_frames, gt_frames, d_preds)
            video_g_loss.backward()
            video_g_optimizer.step()

            if count % 20 == 0:
                save_samples(clips_y, count, "video_real")
                save_samples(video_images, count, "video_fake")

                loss_fp.write(
                    "{},{},{},{}".format(
                        count, video_d_loss_real, video_d_loss_fake, video_g_loss
                    )
                )
                torch.save(video_g_net.state_dict(), "generator_net.pth.tmp")
            count += 1

    loss_fp.close()

    # Final Generator save.
    torch.save(video_g_net.state_dict(), "generator_net.pth")


if __name__ == "__main__":
    main()
