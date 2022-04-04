from scvi import data
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt


from torch import nn
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


class Encoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()


class SoftMinPool(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.requires_grad = False
        self.beta = beta

    def forward(self, x):
        return self.beta * (-(1/self.beta) * torch.log * torch.sum(torch.exp(-self.beta*x)))


class ClusterModule(nn.Module):
    def __init__(self, input_size, n_clusters, beta):
        super().__init__()
        self.beta = beta

        hk = nn.Lienar(input_size, n_clusters, requires_grad=False)
        mc = SoftMinPool(self.beta)

        self.layers = nn.ModuleList(hk, mc)

    def forward(self, z):
        return self.mc(self.hk(z))

    @torch.no_grad()
    def reinit_weights(self, centroid, centroids):
        w_ck = 2 * (centroids - centroid)
        c_norm = torch.linalg.norm(centroid, p=2)
        k_norm = torch.linalg.norm(centroids, p=2, dim=0)

        b_ck = k_norm - c_norm

        self[0].weight = w_ck
        self[0].bias = b_ck


class ClusterModule(nn.Module):
    def __init__(self, input_size, n_clusters, beta):
        super().__init__()
        self.beta = beta
        hk = nn.Lienar(input_size, n_clusters, requires_grad=False)
        mc = SoftMinPool(self.beta)
        self.layers = nn.ModuleList(hk, mc)

    def forward(self, z):
        return self.mc(self.hk(z))

    @torch.no_grad()
    def reinit_weights(self, centroid, centroids):
        w_ck = 2 * (centroids - centroid)
        c_norm = torch.linalg.norm(centroid, p=2)
        k_norm = torch.linalg.norm(centroids, p=2, dim=0)

        b_ck = k_norm - c_norm

        self[0].weight = w_ck
        self[0].bias = b_ck


def create_clusters(predictions, n_clusters):
    centroids = get_centroids(predicitions, n_clusters)
    cluster_net_pool = [ClusterModule(centroid, centroids) for centroid in centroids]
    assignment = [cluster_net(predictions) for cluster_net in cluster_net_pool]
    return assignment


class ACE(nn.Module):
    def __init__(self, n_clusters, encoder_layers, beta):
        super().__init__()
        self.n_clusters = n_clusters
        latent_dimension = encoder_layers[-1]
        self.encoder_decoder = AutoEncoder(encoder_layers)
        self.cluster_net_pool = [ClusterModule(latent_dimension, n_clusters, beta)
                                 for centroid in centroids]

    def forward(self, x):
        z, reconstruction = self.encoder_decoder(x)
        assignment_confidence = [cluster_net(z) for cluster_net in self.cluster_net_pool]
        centroids = get_centroids(z, assignment_confidence)
        self.shift_clusters(centroids)
        return assignment


class CentroidPool(nn.Module):
    def __init__(self, n_clusts, n_dims):
        super().__init__()
        self.coords = torch.normal(mean=0, std=0.01, size=(n_dims, n_clusts), requires_grad=True).T
        self.module_list = [self.coords]
    
    def forward(self, x):
        return torch.cdist(x, self.coords)


def load_data():
    adata = data.heart_cell_atlas_subsampled()
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy() # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="cell_source"
    )
    return adata

adata = load_data()
inp = torch.tensor(adata.X.toarray())
labels = adata.obs['cell_type']
ae = AutoEncoder([adata.n_vars, 256, 2])

dataset = [(inp[i:i+128], torch.tensor(pd.get_dummies(labels[i:i+128]).values)) for i in range(0, len(inp), 128)]
loss_fn = torch.nn.MSELoss()

pool = CentroidPool(11, 2)
params = set()
params |= set(ae.parameters())
params |= set(pool.parameters())
optimizer = torch.optim.Adam(params, lr=3e-4)


def cluster_loss(distances):
    min_dist = torch.min(distances, 1)
#     assignments = torch.where(distances == torch.min(distances, 1), 1, 0)
#     print(assignments)
    return torch.mean(min_dist[0])


for epoch in range(100):
    mean_loss = 0
    for batch in tqdm(dataset):
        X, y = batch
        optimizer.zero_grad()
        output, reconstruction = ae(X)

        recon = loss_fn(reconstruction, X)
        cluster_distances = pool(output)
        clust = cluster_loss(cluster_distances)
        loss = clust + recon
        loss.backward()
        optimizer.step()
        mean_loss += loss

    mean_loss /= len(dataset)
    print(mean_loss.item())


with torch.no_grad():
    output, reconstruction = ae(inp)
    cluster_distances = pool(output)
    predicted_clusters = torch.argmin(cluster_distances, 1)
   
    label_idxs = np.argmax(pd.get_dummies(labels.values).values, axis=-1)

    data_cols = [f'dim_{i}' for i in range(output.shape[1])]
    df = pd.DataFrame(output.numpy(), columns=data_cols)
    df['predicted_cluster'] = predicted_clusters
    df['real_cluster'] = label_idxs


projected = df[data_cols].values
centroids = pool.coords.detach().numpy()

pca = PCA(n_components=2)

pca_transformed_rnaseq = pca.fit_transform(inp)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
sns.scatterplot(x=projected[:,0], y=projected[:,1], hue=pd.Categorical(df['real_cluster']), ax=ax[0], legend=False).set_title('Real cell type')
sns.scatterplot(x=projected[:,0], y=projected[:,1], hue=pd.Categorical(df['predicted_cluster']), ax=ax[1], legend=False).set_title('Predicted cell type')
sns.scatterplot(x=pca_transformed_rnaseq[:,0], y=pca_transformed_rnaseq[:,1], hue=pd.Categorical(df['real_cluster']), ax=ax[2], legend=False).set_title('PCA Transformed Data')
ax[1].scatter(centroids[:,0], centroids[:,1], marker='*', color='black')

plt.show()