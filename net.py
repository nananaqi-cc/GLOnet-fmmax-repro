import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, params, cond_dim=0):
        super().__init__()

        self.noise_dim = int(params.noise_dims)
        self.cond_dim = cond_dim
        self.has_cond = cond_dim > 0

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        input_dim = self.noise_dim + self.cond_dim
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32*16, bias=False),
            nn.BatchNorm1d(32*16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )


    def forward(self, noise, params, weights=None):
        if weights is not None:
            noise = torch.cat([noise, weights], dim=1)
        elif self.has_cond:
            # Zero-pad for backward compat (no conditioning signal)
            padding = torch.zeros(noise.shape[0], self.cond_dim,
                                  device=noise.device, dtype=noise.dtype)
            noise = torch.cat([noise, padding], dim=1)
        net = self.FC(noise)
        net = net.view(-1, 16, 32)
        net = self.CONV(net)
        net = conv1d_meta(net + noise[:, :self.noise_dim].unsqueeze(1), self.gkernel)
        net = torch.tanh(net * params.binary_amp) * 1.05
        return net



