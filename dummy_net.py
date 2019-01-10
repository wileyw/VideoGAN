"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = 2 + x - self.param

        return x
