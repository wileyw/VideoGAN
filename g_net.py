"""
Implemented Vanilla GAN from this Course:

http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

dtype = config.dtype


class Gen1(nn.Module):
    def __init__(self):
        super(Gen1, self).__init__()

        # Generator #1
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(12, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(128, 3, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen2(nn.Module):
    def __init__(self):
        super(Gen2, self).__init__()

        # Generator #2
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(128, 3, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen3(nn.Module):
    def __init__(self):
        super(Gen3, self).__init__()

        # Generator #3
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(512).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)
        self.bn4 = nn.BatchNorm2d(256).type(dtype)

        self.deconv5 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv5.weight)
        self.bn5 = nn.BatchNorm2d(128).type(dtype)

        self.deconv6 = nn.ConvTranspose2d(128, 3, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)
        self.g1.append(self.bn4)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv5)
        self.g1.append(self.bn5)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv6)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen4(nn.Module):
    def __init__(self):
        super(Gen4, self).__init__()

        # Generator #4
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 7, stride=1, padding=3).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 512, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(512).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(512, 256, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)
        self.bn4 = nn.BatchNorm2d(256).type(dtype)

        self.deconv5 = nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv5.weight)
        self.bn5 = nn.BatchNorm2d(128).type(dtype)

        self.deconv6 = nn.ConvTranspose2d(128, 3, 7, stride=1, padding=3).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)
        self.g1.append(self.bn4)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv5)
        self.g1.append(self.bn5)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv6)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class VideoGANGenerator(nn.Module):
    """This class implements the full VideoGAN Generator Network.
    Currently a placeholder that copies the Vanilla GAN Generator network
    """

    def __init__(self):
        super(VideoGANGenerator, self).__init__()

        self.up1 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)
        self.up2 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)
        self.up3 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)

        # Generator #1
        self.g1 = Gen1()
        self.g2 = Gen2()
        self.g3 = Gen3()
        self.g4 = Gen4()

    def forward(self, x):
        out = x.type(dtype)
        h, w = x.shape[-2:]
        # print('x.shape is', x.shape())

        # TODO: Change the image size
        img1 = F.interpolate(out, size=(int(h / 8), int(w / 8)))
        img2 = F.interpolate(out, size=(int(h / 4), int(w / 4)))
        img3 = F.interpolate(out, size=(int(h / 2), int(w / 2)))
        img4 = out

        out = self.g1(img1)
        upsample1 = self.up1(out)
        out = upsample1 + self.g2(torch.cat([img2, upsample1], dim=1))
        upsample2 = self.up2(out)
        out = upsample2 + self.g3(torch.cat([img3, upsample2], dim=1))
        upsample3 = self.up3(out)
        out = upsample3 + self.g4(torch.cat([img4, upsample3], dim=1))

        # Apply tanh at the end
        out = torch.tanh(out)

        return out
