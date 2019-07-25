# Blog Post

## Overview
We reimplemented a Video GAN Neural Network in order to generate synthetic Pacman images. In this project, we explore generating video frames as opposed to generating a single image.

## Motivation
Most common Computer Vision algorithms such as Image Classification, Object Detection, Semantic Segmentation operate primarily on individual images. When we first started this project, we were not familiar with the common techniques for processing video. One of our motivations for starting this project was to learn and become familiar with applying Deep Learning to process video.

In addition to processing videos, we also wanted to take a project that employed GANs (Generative Adversarial Networks). Within Deep Learning, GANs are an ever increasingly popular topic able to generate amazingly high quality image samples.

We wanted to combine both Video Processing and GANs into a single project we're calling VideoGAN.

## Discriminator

## Generator
The generator accepts a history of 4 frames to predict an unknown 5th frame. The history of 4 frames is stacked together as input to the Generator network. Because the input is a series of frames, the generator network structure is an encoder/decoder network. Because the input is a series of frames and not a noise vector, the generator may be more robust to mode collapse.

### Generator Initialization
We initialize our Generator network with xavier initialization. One of the interesting observations using the encoding/decoding network rather than random noise as in the traditional GAN is that the input data is visible in the output image even before doing any optimization.

## Loss Function

## Future Steps
As the number of history steps gets larger, stacking the different images becomes inefficient. As a future step, implementing 3D convolutions where a convolution operation is applied along the temporal dimension is a way to improve video generation.

## Resources
https://arxiv.org/pdf/1511.05440.pdf
