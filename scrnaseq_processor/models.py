import torch
import torch.nn as nn

from typing import List, Optional


class AutoEncoder(nn.Module):
    def __init__(self, encoder_sizes: List[int],
                 decoder_sizes: Optional[List[int]] = None):
        super().__init__()

        encoder_layers = self._build_layers(encoder_sizes)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_sizes = list(reversed(encoder_sizes))
        decoder_layers = self._build_layers(decoder_sizes)
        self.decoder = nn.Sequential(*decoder_layers)

    def _build_layers(self, sizes: List[int]) -> List[nn.Module]:
        num_layers = len(sizes)-1
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        return layers

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)

        return encode, decode


class CentroidPool(nn.Module):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = torch.normal(mean=0, std=0.01, size=(n_dims, n_clusts), requires_grad=True).T
        self.module_list = [self.coords]
    
    def forward(self, x):
        return torch.cdist(x, self.coords)
