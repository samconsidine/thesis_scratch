
import torch
import scanpy as sc
from scrnaseq_processor.models import KMadness
from itertools import product


loss_fn = torch.nn.MSELoss()


def structural_correspondance_loss(D_exp, D_phys, assignments):
    loss = 0.

    return D_phys.sum()

    for k, l in product(range(n_clusts), range(n_clusts)):
        i_mask = assignments == k
        j_mask = assignments == l

        dexp_ij = (D_exp * assignments[:, 0]) * assignments[:, 1].unsqueeze(0).T

        #dexp_ij = D_exp[i_mask][:, j_mask]
        dphys_kl = D_phys[k, l]
        loss += quadratic_loss(dexp_ij, dphys_kl)

    return loss


def quadratic_loss(a, b):
    return 0.5 * ((a - b) ** 2).sum()


class Centroids(torch.nn.Module):
    def __init__(self, n_clusts, n_centroids):
        super().__init__()
        self.coords1 = torch.nn.Parameter(
                torch.randn(n_clusts, n_centroids, requires_grad=True))
        self.coords2 = torch.nn.Parameter(
                torch.randn(n_clusts, n_centroids, requires_grad=True))

    def forward(self, latent):
        return (torch.cdist(latent, self.coords1)).argmin(1)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    n_dims = 2

    data = sc.datasets.paul15()
    n_clusts = 2

    X = torch.randint(0, 10, (100, 3)).float()

    encoder = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], n_dims),
    )

    batches = [X[i:(i+1)] for i in range(0, len(X), 1)]

    centroids = Centroids(n_clusts, n_dims)
    # centroids = KMadness(n_clusts, n_dims)
    dphys = ((centroids.coords1.unsqueeze(1) - centroids.coords2.unsqueeze(0)) ** 2).sum(-1)
    # dphys = centroids.coords
    dexps = [torch.cdist(x, x, p=1) for x in batches]

    #dexp = torch.cdist(X, X, p=1.)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(centroids.parameters()))

    for epoch in range(100):
        for i, batch in enumerate(batches):
            dphys = ((centroids.coords1.unsqueeze(1) - centroids.coords2.unsqueeze(0)) ** 2).sum(-1)
            latent = encoder(batch)
            latent = torch.rand_like(latent)
            dexp = dexps[i]
            dexp = torch.ones_like(dexp)
            assignments = centroids(latent)

            loss = structural_correspondance_loss(dexp, dphys, assignments)
            loss = dphys.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

