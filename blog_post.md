# Blog Post

## Overview
We reimplemented a Video GAN Neural Network in order to generate synthetic Pacman images. In this project, we explore generating video frames as opposed to generating a single image.

## Motivation
Most common Computer Vision algorithms such as Image Classification, Object Detection, Semantic Segmentation operate primarily on individual images. When we first started this project, we were not familiar with the common techniques for processing video. One of our motivations for starting this project was to learn and become familiar with applying Deep Learning to process video.

In addition to processing videos, we also wanted to take a project that employed GANs (Generative Adversarial Networks). Within Deep Learning, GANs are an ever increasingly popular topic able to generate amazingly high quality image samples.

We wanted to combine both Video Processing and GANs into a single project we're calling VideoGAN.

## Dataset
For our dataset, we use frames from the Pacman video game. For our problem, we create a Neural Network that takes a history of N frames to predict an unknown N + 1th frame.

We obtained the Pacman dataset from this github [repository](https://github.com/dyelax/Adversarial_Video_Generation).

Here is the [direct link to the Pacman dataset](https://drive.google.com/open?id=0Byf787GZQ7KvV25xMWpWbV9LdUU).

## Discriminator
The discriminator tries to determine whether an image is real or fake by giving the generated images a score between 0 and 1.

## Generator
The generator accepts a history of 4 frames to predict an unknown 5th frame. The history of 4 frames is stacked together as input to the Generator network. Because the input is a series of frames, the generator network structure is an encoder/decoder network. Because the input is a series of frames and not a noise vector, the generator may be more robust to mode collapse.

### Generator Initialization
We initialize our Generator network with xavier initialization. One of the interesting observations using the encoding/decoding network rather than random noise as in the traditional GAN is that the input data is visible in the output image even before doing any optimization. This observation may suggest that there is a high signal-to-noise ratio in the network meaning the network may be easier to train.

## Loss Function

### Adversarial Loss Function
VideoGAN is a Generative Adversarial Network so we use the Adversarial Loss as one of the loss functions for predicting the next frame. A GAN consists of a Discriminator and a Generator that compete against each other to produce the next frame. The Generator takes a history of N frames to predict the next frame. The Discriminator tries to determine whether the generated frame is real or fake.

### Image Gradient Difference Loss (GDL)
In addition to the Adversarial Loss Function, we implemented the Gradient Difference Loss to sharpen features in the generated image.

## Future Steps
As the number of history steps gets larger, stacking the different images becomes inefficient. As a future step, implementing 3D convolutions where a convolution operation is applied along the temporal dimension is a way to improve video generation.

## Conclusion
In this post, we describe how we trained a Generative Adversarial Network to predict the next frame given a history of past frames.

## Resources
https://arxiv.org/pdf/1511.05440.pdf
