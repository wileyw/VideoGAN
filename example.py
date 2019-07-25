import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_path = "/Users/michael/repo/VideoGAN/Ms_Pacman/Test/0000/00070.png"
        image = cv2.imread(image_path)

        image = np.rollaxis(image, 2, 0)
        image = np.array(image, dtype=np.float32)

        return image


def main():
    dataset = Dataset()
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch in dataset_loader:
        # print(batch.shape)
        torch_image = batch

        # print(torch_image.shape)
        conv1 = nn.Conv2d(3, 3, (2, 1))
        # print(conv1.weight.shape)
        """
        values = np.array([[[[-1., -0., -0.],
        [-0., -1., -0.],
        [-0., -0., -1.]],
       [[ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]]]], dtype=np.float32)
        """
        values = np.array(
            [
                [[[-1], [1]], [[0], [0]], [[0], [0]]],
                [[[0], [0]], [[-1], [1]], [[0], [0]]],
                [[[0], [0]], [[0], [0]], [[-1], [1]]],
            ],
            dtype=np.float32,
        )
        conv1.weight = nn.Parameter(torch.from_numpy(values))

        output = conv1(torch_image)
        output = output.data.numpy()
        # image = np.rollaxis(image, 2, 0)
        output = np.rollaxis(output, 1, 4)

        print(output.shape)

        cv2.imshow("test", output[0])
        cv2.waitKey(0)

        print(conv1.weight)
        print(conv1.weight.shape)


if __name__ == "__main__":
    main()
