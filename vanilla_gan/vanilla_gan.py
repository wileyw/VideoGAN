"""
Implemented Vanilla GAN from this Course:

http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

dtype = config.dtype

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        nin, nout = 3, 32
        self.conv1_depthwise = nn.Conv2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.conv1_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv1_depthwise.weight)
        nn.init.xavier_normal(self.conv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(32).type(dtype)

        nin, nout = 32, 64
        self.conv2_depthwise = nn.Conv2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.conv2_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv2_depthwise.weight)
        nn.init.xavier_normal(self.conv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        nin, nout = 64, 128
        self.conv3_depthwise = nn.Conv2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.conv3_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv3_depthwise.weight)
        nn.init.xavier_normal(self.conv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        nin, nout = 128, 1
        self.conv4_depthwise = nn.Conv2d(nin, nin, 4, stride=1, padding=1, groups=nin).type(dtype)
        self.conv4_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv4_depthwise.weight)
        nn.init.xavier_normal(self.conv4_pointwise.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(dtype)
        # Conv 1
        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Conv 2
        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Conv 3
        out = self.conv3_depthwise(out)
        out = self.conv3_pointwise(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Conv 4
        out = self.conv4_depthwise(out)
        out = self.conv4_pointwise(out)
        if not config.use_wgan_loss:
            out = self.sigmoid(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(100, 128, 4, stride=4, padding=0)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv4.weight)

    def forward(self, x):
        out = self.deconv1(x)
        # TODO: Investigate putting Batch Norm before versus after the RELU layer
        # Resources:
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        # https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be&t=325
        out = self.bn1(out)
        out = F.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.deconv4(out)
        out = torch.tanh(out)

        return out

class GeneratorSkipConnections(nn.Module):
    def make_resblock(self, map_size):
        conv1_depthwise = nn.ConvTranspose2d(map_size, map_size, 3, stride=1, padding=1, groups=map_size).type(dtype)
        conv1_pointwise = nn.ConvTranspose2d(map_size, map_size, 1).type(dtype)
        nn.init.xavier_normal(conv1_depthwise.weight)
        nn.init.xavier_normal(conv1_pointwise.weight)
        bn = nn.BatchNorm2d(map_size).type(dtype)
        conv2_depthwise = nn.ConvTranspose2d(map_size, map_size, 3, stride=1, padding=1, groups=map_size).type(dtype)
        conv2_pointwise = nn.ConvTranspose2d(map_size, map_size, 1).type(dtype)
        nn.init.xavier_normal(conv2_depthwise.weight)
        nn.init.xavier_normal(conv2_pointwise.weight)

        resblock = nn.ModuleList()
        resblock.append(conv1_depthwise)
        resblock.append(bn)
        resblock.append(conv2_pointwise)

        return resblock

    def apply_resblock(self, out, resblock):
        out = resblock[0](out)
        out = resblock[1](out)
        out = F.relu(out)
        out = resblock[2](out)

        return out

    def __init__(self):
        super(GeneratorSkipConnections, self).__init__()

        # TODO: Change convolutions to DepthWise Seperable convolutions
        # TODO: Need to fix Mode Collapse that is occuring in the GAN
        # More info: https://www.quora.com/What-does-it-mean-if-all-produced-images-of-a-GAN-look-the-same

        # Upsampling layer
        nin, nout = 100, 128
        self.deconv1_depthwise = nn.ConvTranspose2d(nin, nin, 4, stride=4, padding=0, groups=nin).type(dtype)
        self.deconv1_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv1_depthwise.weight)
        nn.init.xavier_normal(self.deconv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        # Resnet block
        self.resblock1A = self.make_resblock(128)

        # Upsampling layer
        nin, nout = 128, 64
        self.deconv2_depthwise = nn.ConvTranspose2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.deconv2_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv2_depthwise.weight)
        nn.init.xavier_normal(self.deconv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        # Resnet block
        self.resblock2A = self.make_resblock(64)

        # Upsampling layer 3
        nin, nout = 64, 32
        self.deconv3_depthwise = nn.ConvTranspose2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.deconv3_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv3_depthwise.weight)
        nn.init.xavier_normal(self.deconv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(32).type(dtype)

        # Resnet block
        self.resblock3A = self.make_resblock(32)

        # Upsampling layer 4
        nin, nout = 32, 3
        self.deconv4_depthwise = nn.ConvTranspose2d(nin, nin, 4, stride=2, padding=1, groups=nin).type(dtype)
        self.deconv4_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv4_depthwise.weight)
        nn.init.xavier_normal(self.deconv4_pointwise.weight)

        # Resnet block
        self.resblock4A = self.make_resblock(3)

    def forward(self, x):
        x = x.type(dtype)
        out = x

        # Multi scale image generation seems quite similar to using ResNet skip connections
        # In this case, we only use a single Resnet block instead of the entire Generator so the network is small enough to run on my laptop
        #
        # Upsample 1
        out = self.deconv1_depthwise(out)
        out = self.deconv1_pointwise(out)
        out = self.bn1(out)
        out = upsampled = F.relu(out)

        # Resnet block 1
        out += self.apply_resblock(out.clone(), self.resblock1A)

        # Upsample 2
        out = self.deconv2_depthwise(out)
        out = self.deconv2_pointwise(out)
        out = self.bn2(out)
        out = upsampled = F.relu(out)
        # Resnet block 2
        out += self.apply_resblock(out.clone(), self.resblock2A)

        # Upsample 3
        out = self.deconv3_depthwise(out)
        out = self.deconv3_pointwise(out)
        out = self.bn3(out)
        out = upsampled = F.relu(out)
        # Resnet block 3
        out += self.apply_resblock(out.clone(), self.resblock3A)

        # Upsample 4
        out = self.deconv4_depthwise(out)
        out = self.deconv4_pointwise(out)

        # Resnet block 4
        out += self.apply_resblock(out.clone(), self.resblock4A)

        out = torch.tanh(out)

        return out
