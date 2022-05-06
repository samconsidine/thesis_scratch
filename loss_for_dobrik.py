import torch
import scanpy as sc
from scrnaseq_processor.models import KMadness
from itertools import product


loss_fn = torch.nn.MSELoss()


def structural_correspondance_loss(D_exp, D_phys, assignments):
    loss = 0.

    for k, l in product(range(n_clusts), range(n_clusts)):
        i_mask = ((assignments == k).float() + 1e-6).unsqueeze(1)
        j_mask = ((assignments == l).float() + 1e-6).unsqueeze(0)

        # dexp_ij = (D_exp * assignments[:, 0]) * assignments[:, 1].unsqueeze(0).T

        # dexp_ij = D_exp * i_mask * j_mask
        # breakpoint()
        dexp_ij = (D_exp * assignments[:, k].unsqueeze(1) *
                assignments[:, l].unsqueeze(0))

        # ) + (D_exp * assignments[:, 1].unsqueeze(1))

        #dexp_ij = D_exp[i_mask][:, j_mask]
        dphys_kl = D_phys[k, l]

        loss += quadratic_loss(dexp_ij, dphys_kl)

    return loss


def quadratic_loss(a, b):
    return 0.5 * ((a - b) ** 2).mean()


class Centroids(torch.nn.Module):
    def __init__(self, n_clusts, n_hidden, temperature=10):
        super().__init__()
        self.coords = torch.nn.Parameter(torch.randn(n_clusts, n_hidden, requires_grad=True))
        self.temperature = temperature

    def forward(self, latent):
        return self.temperature * (torch.cdist(latent, self.coords)).softmax(-1)


if __name__ == "__main__":
    n_dims = 32

    data = sc.datasets.paul15()

    # X = torch.randint(0, 10, (2710, 3)).float()
    X = torch.tensor(data.X).float()
    n_clusts = 19

    encoder = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, n_dims)
    )

    batches = [X[i:(i+128)] for i in range(0, len(X), 128)]

    centroids = Centroids(n_clusts, n_dims)
    # centroids = KMadness(n_clusts, n_dims)
    dexps = [torch.cdist(x, x, p=1) for x in batches]

    #optimizer = torch.optim.Adam(list(centroids.parameters()))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(centroids.parameters()), lr=1e-3)
    # optimizer = torch.optim.Adam(list(encoder.parameters()))

    for epoch in range(100):
        loss_batches = 0.
        for i, batch in enumerate(batches):
            latent = encoder(batch)
            dexp = dexps[i]
            assignments = centroids(latent)

            dphys = torch.cdist(centroids.coords, centroids.coords)

            loss = structural_correspondance_loss(dexp, dphys, assignments)
            loss_batches += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss_batches / len(batches))
        print(centroids.coords)

        if epoch % 10 == 0:
            h = encoder(X).detach().numpy()
            x = h[:, 0]
            y = h[:, 1]
            import seaborn as sns
            sns.scatterplot(x=x, y=y, hue=data.obs['paul15_clusters'].values)
            coords = centroids.coords.detach().numpy()
            cx = coords[:, 0]
            cy = coords[:, 1]
            sns.scatterplot(x=cx, y=cy, marker='+')
            import matplotlib.pyplot as plt
            plt.show()

