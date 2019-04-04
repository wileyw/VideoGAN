# VideoGAN
Implement video generative model

Original Paper:
https://arxiv.org/pdf/1511.05440.pdf

Pacman dataset:
https://drive.google.com/open?id=0Byf787GZQ7KvV25xMWpWbV9LdUU

Adversarial Video Generation:
https://github.com/dyelax/Adversarial_Video_Generation

## Generate VideoGAN Data
The VideoGAN training data requires preprocessing. To generate the VideoGAN data:

1. Download the Ms_Pacman dataset from https://drive.google.com/open?id=0Byf787GZQ7KvV25xMWpWbV9LdUU
2. Unzip Ms_Pacman dataset in `VideoGAN/Ms_Pacman`
3. Run the following commands in `VideoGAN`
```
mkdir train
python process_data.py Ms_Pacman/Train train
```

# Instructions to run Vanilla GAN
```
bash vanilla_gan/download_dataset.sh
cp -r a4-code-v2-updated/emojis emojis
python3 process.py
```

# Tips and tricks to train GANs
https://github.com/soumith/ganhacks

# TODO
1. Generate the entire Pacman board from the generative network
2. Evaluate predicted frames using Peak Signal to Noise Ratio (PSNR), Structural
Similarity Index Measure (SSIM), [Inception Distance](https://nealjean.com/ml/frechet-inception-distance/)
3. Change loss function to use log() function
