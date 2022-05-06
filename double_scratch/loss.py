import torch
import scanpy as sc
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns


loss_fn = torch.nn.MSELoss()


def structural_correspondance_loss(D_exp, D_phys, assignments):
    loss = 0.

    for k, l in product(range(n_clusts), range(n_clusts)):
        i_mask = assignments == k
        j_mask = assignments == l

        dexp_ij = D_exp[i_mask][:, j_mask]
        dphys_kl = D_phys[k, l]
        loss += quadratic_loss(dexp_ij, dphys_kl)

    return loss


def structural_correspondance_loss_gradient_test(D_exp, D_phys, assignments):
    """ Just test whether D_phys and D_exp are tracked wrt the encoder gradient """
    loss = 0.
    clusters_vec = D_phys[assignments]  # for good measure

    return loss_fn(dexp.sum(0), dphys.sum(0))

    for k, l in product(range(n_clusts), range(n_clusts)):
        i_mask = assignments == k
        j_mask = assignments == l

        dexp_ij = D_exp
        dphys_kl = D_phys[k, l]
        loss += loss_fn(dexp_ij, dphys_kl)

    return loss


def quadratic_loss(a, b):
    return 0.5 * ((a - b) ** 2).sum()


class Centroids(torch.nn.Module):
    def __init__(self, n_clusts, n_centroids):
        super().__init__()
        self.coords = torch.nn.Parameter(
                torch.randn(n_clusts, n_centroids, requires_grad=True))

    def forward(self, latent):
        return (torch.cdist(latent, self.coords)).min(1)[1]


if __name__ == "__main__":
    n_dims = 2
    data = sc.datasets.paul15()
    n_clusts = data.obs['paul15_clusters'].unique().shape[0]
    X = torch.tensor(data.X)

    encoder = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], n_dims),
    )
    batches = [X[i:(i+128)] for i in range(0, len(X), 128)]

    centroids = Centroids(n_clusts, n_dims)
    dphys = torch.cdist(centroids.coords.detach(), centroids.coords.detach())
    dexps = [torch.cdist(x, x, p=1) for x in batches]

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(centroids.parameters()))

    for epoch in range(100):
        for i, batch in enumerate(batches):
            latent = encoder(batch)
            dexp = dexps[i]
            assignments = centroids(latent)

            loss = structural_correspondance_loss(dexp, dphys, assignments)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        latent = encoder(X)
        print(latent.shape)

        dexp = torch.cdist(latent, latent, p=1.0)

        assignments = centroids(latent)

        preds = latent.detach().numpy()
        x = preds[:, 0]
        y = preds[:, 1]

        cents = centroids.detach().numpy()
        cx = cents[:, 0]
        cy = cents[:, 1]

        print(loss)
        sns.scatterplot(x=x, y=y, hue=data.obs['paul15_clusters'].values)
        sns.scatterplot(x=cx, y=cy, marker="+")
        plt.show()


