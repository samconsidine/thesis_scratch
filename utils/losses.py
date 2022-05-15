from __future__ import annotations

import torch
from torch.linalg import vector_norm
from torch import Tensor
from torch.nn import MSELoss, Module
from dataclasses import dataclass

from typing import Tuple, List


def lineseg_projection_singletons(p1, p2, p3):
    l2 = torch.sum((p1-p2)**2)
    # t = torch.sum((p3 - p1) * (p2 - p1)) / l2
    t = torch.clamp(torch.sum((p3 - p1) * (p2 - p1)) / l2, min=0, max=1)
    projection = p1 + t * (p2 - p1)
    return projection


def lineseg_projection_oneseg(p1, p2, p3):
    """ 
    p1 shape = 1 x d 
    p2 shape = 1 x d
    p3 shape = n x d
    """
    l2 = torch.sum((p1-p2)**2)
    p1p2 = p2 - p1  # 1 x d
    p1p3 = p3 - p1  # n x d

    t = torch.sum((p3 - p1) * (p2 - p1), dim=-1) / l2  # n
    projection = p1.expand(p3.shape[0], -1) + t.unsqueeze(-1) * ((p2 - p1).expand(p3.shape[0], -1))
    #    n x d   n x d                        n x 1              n x d   
    return projection


def lineseg_projection(p1, p2, p3):
    """
    s - num segments
    t shape = n x s
    p1 shape = s x d
    p2 shape = s x d
    p3 shape = n x d

    output = n x s x d
    """
    l2 = torch.sum((p1 - p2)**2, dim=-1) # s
    p1p2 = p2 - p1  # s x d
    p1p3 = p3.unsqueeze(1) - p1.unsqueeze(0)  # n x 1 x d - 1 x s x d -> n x s x d
    
    t = torch.sum(p1p2 * p1p3, dim=-1) / l2 # n x s
    projection  =  p1 + t.unsqueeze(-1) * p1p2
    # n x s x d =  s x d     n x s x d
    return projection


def test_linesegs():
    p1 = torch.rand(500, 64)
    p2 = torch.rand(500, 64)
    p3 = torch.randn(1000, 64)

    full_proj = lineseg_projection(p1, p2, p3)

    for seg in range(500):
        assert torch.allclose(full_proj[:, seg, :], 
               lineseg_projection_oneseg(p1[seg], p2[seg], p3))
    print("All's good")


def project_onto_mst(
        latent: Tensor,  # n x d
        tree: MST
    ) -> Tuple[Tensor, Tensor]:

    # distances = torch.cdist(latent, tree.midpoints, p=2.0)
    # projection_probabilities = distances.softmax(1)

    # projected_coords = tree.midpoints.unsqueeze(1).expand(-1, latent.shape[0], -1) 

    edges = tree.edges[:, tree.edges[0] != tree.edges[1]]

    from_segments = tree.nodes[edges[0]]
    to_segments = tree.nodes[edges[1]]
    projection_coords = lineseg_projection(from_segments, to_segments, latent)  # n x s x d

    # distances = torch.cdist(latent.unsqueeze(1), projection_coords)
    # n x s   =             n x d                n x s x d 
    distances = vector_norm(latent.unsqueeze(1).broadcast_to(projection_coords.shape) - 
                            projection_coords, dim=-1)

    projection_probabilities = distances.softmax(1)

    return projection_probabilities, projection_coords


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

    reconstructions = decoder(projected_coords)
    distances = ((X.unsqueeze(1) - reconstructions).pow(2)).sum(-1).sqrt()
    loss = distances * (mst.probabilities.view(-1).unsqueeze(0) 
                        * projection_probabilities).T

    return loss.mean()


if __name__ == "__main__":
    test_linesegs()
    exit()

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


