import torch

use_wgan_loss = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor
