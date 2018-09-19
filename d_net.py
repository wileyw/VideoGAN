"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import resize

# Default settings of discriminator networks.
IMG_H = 32
IMG_W = 32

# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [
    [3, 64],
    [3, 64, 128, 128],
    [3, 128, 256, 256],
    [3, 128, 256, 512, 128],
]
# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [
    [3],
    [3, 3, 3],
    [5, 5, 5],
    [7, 7, 5, 5],
]
# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [
    [512, 256, 1],
    [1024, 512, 1],
    [1024, 512, 1],
    [1024, 512, 1],
]


class DScaleNet(nn.Module):
    def __init__(self,
                 img_height,
                 img_width,
                 kernel_sizes,
                 conv_layer_fms,
                 scale_fc_layer_sizes):
        super(DScaleNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Define conv blocks.
        for i in range(len(kernel_sizes)):
            self.conv_layers.append(nn.Conv2d(conv_layer_fms[i],
                                              conv_layer_fms[i+1],
                                              kernel_sizes[i]))
            img_height -= kernel_sizes[i] - 1
            img_width -= kernel_sizes[i] - 1

        # Define fc blocks.
        # Add fc layer to convert max_pool layer to compatible initial flat size.
        img_height /= 2
        img_width /= 2
        scale_fc_layer_sizes.insert(
            0, int(img_height * img_width * conv_layer_fms[-1]))
        for i in range(len(scale_fc_layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(scale_fc_layer_sizes[i],
                                            scale_fc_layer_sizes[i+1]))

    def forward(self, x):
        # Run convolutions.
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        # Max pooling layer.
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size()[0], -1)

        # Run fc.
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        x = F.sigmoid(self.fc_layers[-1](x))
        x = F.clamp(x, 0.1, 0.9)

        return x


class DiscriminatorModel(nn.Module):
    def __init__(self,
                 img_height=IMG_H, img_width=IMG_W,
                 kernel_sizes_list=SCALE_KERNEL_SIZES_D,
                 conv_layer_fms_list=SCALE_CONV_FMS_D,
                 scale_fc_layer_sizes_list=SCALE_FC_LAYER_SIZES_D):
        super(DiscriminatorModel, self).__init__()

        self.scale_nets = nn.ModuleList()
        self.img_dims = []
        zip_data = zip(kernel_sizes_list,
                       conv_layer_fms_list,
                       scale_fc_layer_sizes_list)
        for scale_num, data in enumerate(zip_data):
            kernel_sizes, conv_layer_fms, scale_fc_layer_sizes = data
            scale_factor = 1. / 2 ** ((len(kernel_sizes_list) - 1) - scale_num)
            img_dim = (int(img_height * scale_factor),
                       int(img_width * scale_factor))

            self.img_dims.append(img_dim)
            self.scale_nets.append(
                DScaleNet(img_dim[0],
                          img_dim[1],
                          kernel_sizes,
                          conv_layer_fms,
                          scale_fc_layer_sizes))

    def forward(self, x):
        out = []
        for img_dim, scale_net in zip(self.img_dims, self.scale_nets):
            scale_net_x = resize(x, img_dim)
            out.append(scale_net.forward(x))
        return torch.stack(out)
