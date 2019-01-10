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

    g_optimizer = optim.SGD(G.parameters(), lr=0.001, momentum=0.9)

    print('Dummy Parameters', list(Dummy.parameters()))
    dummy_optimizer = optim.Adam(Dummy.parameters(), lr=0.01)

    g_optimizer.zero_grad()

    param_values = []
    for epoch in range(1000):
        for i_batch, sample_batch in enumerate(dataset_loader):
            before_batch = sample_batch['image0'].float()
            after_batch = sample_batch['image1'].float()

            # Clear gradients before calling loss.backward() and optimizer.step()
            dummy_optimizer.zero_grad()

            #simple_loss = (generated_image - dog_data).pow(2)
            #simple_loss = simple_loss.sum(1).sum(1).sum(1)

            simple_loss = Dummy(-5).pow(2)
            print('Simple Loss:', simple_loss)
            param = list(Dummy.parameters())[0].tolist()[0]
            print('Solution:', param)

            param_values.append(param)

            simple_loss.backward()
            dummy_optimizer.step()

            #print('Size of Generator Input:', before_batch.shape)
            #generated_image = G(before_batch)
            #save_dummy_data(generated_image, epoch)

            break

    print(param_values)
    matplotlib.pyplot.plot(param_values)
    plt.show()


if __name__ == '__main__':
    main()
