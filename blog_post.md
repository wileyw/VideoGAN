# Blog Post

## Overview
We reimplemented a Video GAN Neural Network in order to generate synthetic Pacman images. In this project, we explore generating video frames as opposed to generating a single image.

## Discriminator

## Generator
The generator accepts a history of 4 frames to generate an unknown 5th frame. The history of 4 frames is stacked together as input to the Generator network. Because the input is a series of frames, the generator network structure is an encoder/decoder network.

### Generator Initialization
We initialize our Generator network with xavier initialization. One of the interesting observations using the encoding/decoding network rather than random noise as in the traditional GAN is that the input data is visible in the output image even before doing any optimization.

## Loss Function

## Resources
https://arxiv.org/pdf/1511.05440.pdf
