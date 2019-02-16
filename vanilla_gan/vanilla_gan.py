"""
Implemented Vanilla GAN from this Course:

http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 1, 4, stride=4, padding=1)
        nn.init.xavier_normal(self.conv4.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Conv 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Conv 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Conv 4
        out = self.conv4(out)
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
