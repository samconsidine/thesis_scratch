import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid, ReLU, Sequential

from typing import List, Optional


class AutoEncoder(nn.Module):
    def __init__(self, encoder_sizes: List[int],
                 decoder_sizes: Optional[List[int]] = None):
        super().__init__()

        encoder_layers = self._build_layers(encoder_sizes)
        # self.encoder = nn.Sequential(*encoder_layers)

        decoder_sizes = list(reversed(encoder_sizes))
        decoder_layers = self._build_layers(decoder_sizes)
        # self.decoder = nn.Sequential(*decoder_layers)

        self.encoder = Sequential(
            Linear(encoder_sizes[0], 128),
            ReLU(),
            Linear(128, 2),
            Sigmoid()
        )
        self.decoder = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, encoder_sizes[0])
        )

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
        # self.coords = nn.Parameter(torch.normal(mean=1, std=1, size=(n_dims, n_clusts), requires_grad=True).T)
        self.coords = nn.Parameter(torch.rand(n_clusts, n_dims, requires_grad=True))
        self.module_list = [self.coords]
        self.n_clusts = n_clusts
    
    def forward(self, latent):
        closest_centroid = torch.cdist(latent, self.coords).min(1)[1]
        print(closest_centroid.shape)
        cluster_distances = torch.linalg.norm((latent - self.coords[closest_centroid]), dim=1)
        return closest_centroid


class KMadness(nn.Module):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = nn.Parameter(torch.rand(size=(n_clusts, n_dims)))

    @property
    def w(self):
        return f1(self.coords).T

    @property
    def b(self):
        return f2(sqnorm(self.coords))

    def fc(self, h):
        return (-(-h).topk(2, 1)[0])[:, 1, :]

    def forward(self, x):
        h = torch.matmul(self.w, x.T).T + self.b
        cluster_assignments = self.fc(h) > 0
        return cluster_assignments

    def closest_centroids(self, assignments):
        return self.coords[assignments.max(-1)[1]]


def f1(tensor):
    tensor = tensor.permute(1, 0)
    return torch.sub(tensor.unsqueeze(dim=2), tensor.unsqueeze(dim=1))


def f2(tensor):
    return tensor.unsqueeze(0).T - tensor


def sqnorm(tensor):
    return torch.linalg.norm(tensor, dim=1)**2
