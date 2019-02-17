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

class GeneratorSkipConnections(nn.Module):
    def __init__(self):
        super(GeneratorSkipConnections, self).__init__()

        # Upsampling layer
        self.deconv1 = nn.ConvTranspose2d(100, 128, 4, stride=4, padding=0)
        nn.init.xavier_normal(self.deconv1.weight)

        # Resnet block
        self.deconv1A = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv1A.weight)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv1B = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv1B.weight)

        # Upsampling layer
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv2.weight)

        # Resnet block
        self.deconv2A = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv2A.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2B = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv2B.weight)

        # Upsampling layer 3
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv3.weight)

        # Resnet block
        self.deconv3A = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv3A.weight)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3B = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv3B.weight)

        # Upsampling layer 4
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        nn.init.xavier_normal(self.deconv4.weight)

        # Resnet block
        self.deconv4A = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv4A.weight)
        self.bn4 = nn.BatchNorm2d(3)
        self.deconv4B = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
        nn.init.xavier_normal(self.deconv4B.weight)

    def forward(self, x):
        out = x

        # Multi scale image generation seems quite similar to using ResNet skip connections
        # In this case, we only use a single Resnet block instead of the entire Generator so the network is small enough to run on my laptop
        #
        # Upsample 1
        out = upsampled = self.deconv1(out)
        # Resnet block 1
        out = self.deconv1A(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.deconv1B(out)
        out = upsampled + out

        # Upsample 2
        out = upsampled = self.deconv2(out)
        # Resnet block 2
        out = self.deconv2A(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.deconv2B(out)
        out = upsampled + out

        # Upsample 3
        out = upsampled = self.deconv3(out)
        # Resnet block 3
        out = self.deconv3A(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.deconv3B(out)
        out = upsampled + out

        # Upsample 4
        out = upsampled = self.deconv4(out)

        # Resnet block 4
        out = self.deconv4A(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.deconv4B(out)
        out = upsampled + out

        out = torch.tanh(out)

        return out
