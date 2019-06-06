import torch

use_wgan_loss = False

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
