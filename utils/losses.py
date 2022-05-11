from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import MSELoss, Module
from dataclasses import dataclass

from typing import Tuple, List


def project_onto_mst(
        latent: Tensor, 
        tree: MST
    ) -> Tuple[Tensor, Tensor]:

    distances = torch.cdist(latent, tree.midpoints, p=2.0)
    projection_probabilities = distances.softmax(1)

    projected_coords = tree.midpoints.unsqueeze(1).expand(-1, latent.shape[0], -1) 

    return projection_probabilities, projected_coords


class Centroids(torch.nn.Module):
    def __init__(self, n_clusts, dim):
        super().__init__()
        self.coords = torch.nn.Parameter(torch.randn(n_clusts, dim, requires_grad=True))


def mst_reconstruction_loss(
        latent: Tensor,  # Output of encoder
        mst: MST,   # Tensor of edge_index
        X: Tensor,
        decoder: Module
    ) -> float:

    projection_probabilities, projected_coords = project_onto_mst(latent, mst)
    # fake_loss = (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities).min(-1).values.sum()
    # return fake_loss

    reconstructions = decoder(projected_coords)
    distances = ((X - reconstructions)**2).sum(-1).sqrt()
    loss = distances * (mst.probabilities.view(-1).unsqueeze(0) * projection_probabilities).T

    return loss.mean()


if __name__ == "__main__":
    import scanpy as sc
    latent_dim = 2
    data = sc.datasets.paul15()
    X = torch.tensor(data.X).float()

    encoder = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, latent_dim)
    )

    decoder = torch.nn.Sequential(
        torch.nn.Linear(2, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, X.shape[1])
    )
    
    centroids = Centroids(4, 2)
    optimizer = torch.optim.Adam(list(centroids.parameters()) +
            list(encoder.parameters()) + list(decoder.parameters()))

    n_epochs = 50
    for epoch in range(n_epochs):

        latent = encoder(X)
        mst = MST(
            nodes=centroids.coords,
            edges=torch.tensor([[0, 1, 2], [1, 2, 3]])  # This
        )

        loss = mst_reconstruction_loss(latent, mst, X, decoder)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

        if epoch % (n_epochs - 1) == 0:
            with torch.no_grad():
                import seaborn as sns
                import matplotlib.pyplot as plt

                outs = encoder(X).detach().numpy()
                xs = outs[:, 0]
                ys = outs[:, 1]

                coords = centroids.coords
                cx = coords[:, 0]
                cy = coords[:, 1]

                sns.scatterplot(x=xs, y=ys, hue=data.obs['paul15_clusters'].values)
                sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
                plt.show()


