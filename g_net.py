"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default settings of discriminator networks.
IMG_H = 32
IMG_W = 32

HIST_LEN = 1

# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_G = [
    [3 * (HIST_LEN), 128, 256, 128, 3],
    [3 * (HIST_LEN), 128, 256, 128, 3],
    [3 * (HIST_LEN), 128, 256, 512, 256, 128, 3],
    [3 * (HIST_LEN), 128, 256, 512, 256, 128, 3]
]

# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_G = [
    [3, 3, 3, 3],
    [5, 3, 3, 5],
    [5, 3, 3, 3, 3, 5],
    [7, 5, 5, 5, 5, 7],
]


class GScaleNet(nn.Module):
    def __init__(self,
                 img_height,
                 img_width,
                 kernel_sizes,
                 conv_layer_fms):
        super(GScaleNet, self).__init__()

        self.conv_layers = nn.ModuleList()

        # Define conv blocks.
        for i in range(len(kernel_sizes)):
            before_size = conv_layer_fms[i]
            after_size = conv_layer_fms[i + 1]
            if before_size < after_size:
                layer = nn.Conv2d(before_size, after_size, kernel_sizes[i], stride=2)
            else:
                layer = nn.ConvTranspose2d(before_size, after_size, kernel_sizes[i], stride=2)
            self.conv_layers.append(layer)

    def forward(self, x):
        # Run convolutions.
        print(x.size())
        for layer in self.conv_layers:
            print(layer)
            x = F.relu(layer(x))
            print(x.size())

        return x


class GeneratorModel(nn.Module):
    def __init__(self,
                 img_height=IMG_H, img_width=IMG_W,
                 kernel_sizes_list=SCALE_KERNEL_SIZES_G,
                 conv_layer_fms_list=SCALE_CONV_G):
        super(GeneratorModel, self).__init__()

        self.scale_nets = nn.ModuleList()
        self.img_dims = []
        zip_data = zip(kernel_sizes_list,
                       conv_layer_fms_list)
        for scale_num, data in enumerate(zip_data):
            kernel_sizes, conv_layer_fms = data
            #scale_factor = 1. / 2 ** ((len(kernel_sizes_list) - 1) - scale_num)
            scale_factor = 1
            img_dim = (int(img_height * scale_factor),
                       int(img_width * scale_factor))

            self.img_dims.append(img_dim)
            self.scale_nets.append(
                GScaleNet(img_dim[0],
                          img_dim[1],
                          kernel_sizes,
                          conv_layer_fms))

    def forward(self, x):
        out = []
        x = F.interpolate(x, size=(32, 32))
        for img_dim, scale_net in zip(self.img_dims, self.scale_nets):
            scale_net_x = F.interpolate(x, size=img_dim)
            out.append(scale_net.forward(scale_net_x))

        return torch.stack(out)
