from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear, BatchNorm1d, LeakyReLU, ReLU
import scanpy as sc
from dataclasses import dataclass

from typing import List


@dataclass
class AutoEncoderConfig:
    n_features: int
    emb_dim: int
    train_epochs: int
    lambda_d: float
    lambda_c: float
    lr: float
    batch_size: int
    layer_sizes: List[int]


@dataclass
class AutoEncoder:
    encoder: Encoder
    cluster: Clusterer
    decoder: Decoder
    config: AutoEncoderConfig

    def cluster_gene_expr(self, X):
        self.eval()
        return self.encoder(self.cluster(X))

    @classmethod
    def from_dataset(cls, X, config):
        ...

    @classmethod
    def from_saved(cls, fp, config):
        ...

    def eval(self):
        encoder.eval()
        cluster.eval()
        decoder.eval()


class Encoder(Module):
    """
    Encoding layers.
    """
    def __init__(self, n_features, emb_dim, layers):
        super().__init__()

        self.net = Sequential(
            Linear(n_features, layers[0]),
            BatchNorm1d(layers[0]),
            LeakyReLU(negative_slope=0.2),
            Linear(layers[0], layers[1]),
            BatchNorm1d(layers[1]),
            LeakyReLU(negative_slope=0.2),
            Linear(layers[1], layers[2]),
            BatchNorm1d(layers[2]),
            LeakyReLU(negative_slope=0.2),
            Linear(layers[2], emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class Cluster(Module):
    """
    Decoding and clustering layers.
    """
    def __init__(self, emb_dim, layers):
        super().__init__()
        self.net = Sequential(
            Linear(emb_dim, layers[2]),
            BatchNorm1d(layers[2]),
            LeakyReLU(negative_slope=0.2),
            Linear(layers[2], layers[1]),
            BatchNorm1d(layers[1]),
            LeakyReLU(negative_slope=0.2),
            Linear(layers[1], layers[0]),
            BatchNorm1d(layers[0]), ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    """
    Reconstructing layer.
    """
    def __init__(self, n_features, layers):
        super().__init__()
        modules = [Linear(layers[0], n_features)]
        self.net = Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def gather_clusters(acts, binmin=10, max_clusts=19, merge_k_nearest=1):
    acts = acts / acts.max()
    binarised = (acts > 1e-6).float()
    unique_rows, counts = torch.unique(binarised, dim=0, return_counts=True)
    unique_rows = unique_rows[counts > binmin]
    n_clusters = unique_rows.shape[0]

    n_clusters = 0
    clusters = -1 * torch.ones(acts.shape[0])
    for i, row in enumerate(unique_rows):
        rows_equal_to_this_code = torch.where(torch.all(binarised == row, axis=1))[0]

        clusters[rows_equal_to_this_code] = n_clusters
        n_clusters += 1

    breakpoint()
    
    return binarised.argmax(1)


def reg_c(act):
    """
    ID (information dimension) regularization (minimizing Shannon entropy)
    """
    p = torch.sum(act, 0, keepdim=True)
    normalized = p / torch.sum(p)
    return torch.sum(-normalized * torch.log(normalized + 1e-9))


def reg_d(act, x):
    """
    Intra-cluster distances regularization.
    """
    act = act / torch.max(act)
    dist = pairwise_dist(act, act)
    same_cluster_probs = gaussian_kernel_matrix(dist)
    same_cluster_probs = same_cluster_probs - torch.min(same_cluster_probs)
    same_cluster_probs = same_cluster_probs / torch.max(same_cluster_probs)
    original_dist =  pairwise_dist(x, x)
    original_dist = torch.sqrt(original_dist + 1e-3)
    intracluster_dist = torch.matmul(original_dist, same_cluster_probs)
    return torch.mean(intracluster_dist)


def batch(X, batch_size):
    perm = torch.randperm(X.shape[0])

    for i in range(0, X.shape[0], batch_size):
        yield X[perm[i:(i+batch_size)]]


def pairwise_dist(x1, x2):
    r1 = torch.sum(x1*x1, 1, keepdim=True)
    r2 = torch.sum(x2*x2, 1, keepdim=True)
    K = r1 - 2*torch.matmul(x1, torch.t(x2)) + torch.t(r2)
    return K


def gaussian_kernel_matrix(dist):
    # Multi-scale RBF kernel. (average of some bandwidths of gaussian kernels)
    # This must be properly set for the scale of embedding space
    sigmas = [1e-5, 1e-4, 1e-3, 1e-2]
    beta = 1. / (2. * torch.unsqueeze(torch.tensor(sigmas), 1))
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    return torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape) / len(sigmas)


def join_params(*modules):
    return sum([list(m.parameters()) for m in modules], start=[])

def train(X, config):
    encoder = Encoder(config.n_features, config.emb_dim, config.layer_sizes)
    cluster = Cluster(config.emb_dim, config.layer_sizes)
    decoder = Decoder(config.n_features, config.layer_sizes)
    
    reconstruction_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(join_params(encoder, cluster, decoder), lr=config.lr)

    for epoch in range(config.train_epochs):
        losses = []
        for x in batch(X, batch_size=config.batch_size):
            embed = encoder(x)
            cl = cluster(embed)
            recon = decoder(cl)

            recon_loss = reconstruction_loss(recon, x)

            id_loss = reg_c(cl)
            intradist_loss = reg_d(cl, x)

            loss = (recon_loss 
                    + config.lambda_c * id_loss 
                    + config.lambda_d * intradist_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(gather_clusters(cl))

        print(f'Epoch {epoch} finished. Loss={sum(losses)}')
    return AutoEncoder(encoder=encoder, cluster=cluster, decoder=decoder, config=config)


def preprocess(X):
    return torch.asinh(X / 5.)

if __name__ == "__main__":
    data = sc.datasets.paul15()
    X = torch.tensor(data.X, dtype=torch.float32)
    # X = preprocess(X)
    config = AutoEncoderConfig(
        n_features=X.shape[1],
        emb_dim=2,
        train_epochs=100,
        layer_sizes=[256, 128, 32],
        lambda_c=0.02,
        lambda_d=0.04,
        lr=0.001,
        batch_size=128,
    )
    ae = train(X, config)

    import seaborn as sns
    import matplotlib.pyplot as plt

    encoded = ae.encoder(X).detach().numpy()
    xs = encoded[:, 0]
    ys = encoded[:, 1]
    sns.scatterplot(x=xs, y=ys, hue=data.obs[data.obs.columns[0]])
    plt.show()

