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
        self.conv1_depthwise = nn.Conv2d(nin, nout, 4, stride=2, padding=1, groups=1).type(dtype)
        #self.conv1_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv1_depthwise.weight)
        #nn.init.xavier_normal(self.conv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(32).type(dtype)

        nin, nout = 32, 64
        self.conv2_depthwise = nn.Conv2d(nin, nout, 4, stride=2, padding=1, groups=1).type(dtype)
        #self.conv2_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv2_depthwise.weight)
        #nn.init.xavier_normal(self.conv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        nin, nout = 64, 128
        self.conv3_depthwise = nn.Conv2d(nin, nout, 4, stride=2, padding=1, groups=1).type(dtype)
        #self.conv3_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv3_depthwise.weight)
        #nn.init.xavier_normal(self.conv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        nin, nout = 128, 1
        self.conv4_depthwise = nn.Conv2d(nin, nout, 4, stride=1, padding=1, groups=1).type(dtype)
        #self.conv4_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv4_depthwise.weight)
        #nn.init.xavier_normal(self.conv4_pointwise.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(dtype)
        # Conv 1
        out = self.conv1_depthwise(x)
        #out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Conv 2
        out = self.conv2_depthwise(out)
        #out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Conv 3
        out = self.conv3_depthwise(out)
        #out = self.conv3_pointwise(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Conv 4
        out = self.conv4_depthwise(out)
        #out = self.conv4_pointwise(out)
        if not config.use_wgan_loss:
            out = self.sigmoid(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(12, 128, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)
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
