"""
# padding for convolutions in the discriminator model
PADDING_D = 'VALID'
# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [[3, 64],
                    [3, 64, 128, 128],
                    [3, 128, 256, 256],
                    [3, 128, 256, 512, 128]]
# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [[3],
                        [3, 3, 3],
                        [5, 5, 5],
                        [7, 7, 5, 5]]
# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [[512, 256, 1],
                          [1024, 512, 1],
                          [1024, 512, 1],
                          [1024, 512, 1]]

NUM_SCALE_NETS_D = 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DScaleNet(nn.Module):
    def __init__(self, kernel_sizes, conv_layer_fms, scale_fc_layer_sizes):
        super(DScaleNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Define conv blocks.
        for i in range(len(kernel_sizes)):
            self.conv_layers.append(nn.Conv2d(conv_layer_fms[i],
                                              conv_layer_fms[i+1],
                                              kernel_sizes[i]))

        # Define fc blocks.
        for i in range(len(scale_fc_layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(scale_fc_layer_sizes[i],
                                            scale_fc_layer_sizes[i+1]))

    def forward(self, x):
        # Run convolutions.
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        # Max pooling layer.
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Run fc.
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
        x = F.sigmoid(self.fc_layers[-1](x))
        x = F.clamp(x, 0.1, 0.9)

        return x


class DiscriminatorModel(nn.Module):
    def __init__(self,
                 kernel_sizes_list,
                 conv_layer_fms_list,
                 scale_fc_layer_sizes_list):
        super(DiscriminatorModel, self).__init__()

        self.scale_nets = nn.ModuleList()
        for (kernel_sizes,
             conv_layer_fms,
             scale_fc_layer_sizes) in zip(kernel_sizes_list,
                                          conv_layer_fms_list,
                                          scale_fc_layer_sizes_list):
             self.scale_nets.append(DScaleNet(kernel_sizes,
                                              conv_layer_fms,
                                              scale_fc_layer_sizes))

    def forward(self, x):
        return [scale_net(x) for scale_net in self.scale_nets]
