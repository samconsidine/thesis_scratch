import torch
import torch.nn as nn


class CentroidPool(nn.Module):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = torch.normal(mean=0, std=0.01, size=(n_dims, n_clusts), requires_grad=True).T
        self.module_list = [self.coords]

    def forward(self, x):
        return torch.cdist(x, self.coords)



pool = CentroidPool(11, 2)
params = set()
params |= set(ae.parameters())
params |= set(pool.parameters())
optimizer = torch.optim.Adam(params, lr=3e-4)
