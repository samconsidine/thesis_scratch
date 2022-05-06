import torch
from torch import Tensor
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from scrnaseq_processor.models import AutoEncoder, CentroidPool
from scrnaseq_processor.data import load_data, process_for_model, batch_data


def combine_params(*models):
    params = set()
    for model in models:
        params |= set(model.parameters())
    return params


def train(dataset, ae, pool, epochs, clust_coef=0.01, recon_coef=1.):
    # params = combine_params(ae, pool)
    # optimizer = torch.optim.Adam(params, lr=3e-4)
    loss_fn = nn.MSELoss()

    params = set()
    params |= set(ae.parameters())
    params |= set(pool.parameters())

    optimizer = torch.optim.Adam(params, lr=3e-4)

    for epoch in range(epochs):
        mean_loss = 0
        cluster_loss_ = 0
        recon_loss_ = 0
        for batch in tqdm(dataset):
            X, y = batch
            optimizer.zero_grad()
            latent, reconstruction = ae(X)

            # print(reconstruction.shape, X.shape)
            # print(X)
            # print(reconstruction)
            recon = loss_fn(reconstruction, X)
            # centroids = pool.closest_centroids(pool(latent))
            centroids = pool.coords
            closest_centroid = torch.cdist(latent, centroids).min(1)[1]
            cluster_distances = torch.mean(torch.linalg.norm((latent - centroids[closest_centroid]), dim=1))
            clust = cluster_distances# cluster_loss(cluster_distances)
            loss = clust_coef * clust + recon_coef * recon
            # loss = recon
            loss.backward()
            optimizer.step()
            mean_loss += loss
            recon_loss_ += recon * recon_coef
            cluster_loss_ += clust * clust_coef

        mean_loss /= len(dataset)
        cluster_loss_ /= len(dataset)
        recon_loss_ /= len(dataset)
        print(f'{mean_loss.item()=}')
        print(f'{recon_loss_.item()=}')
        print(f'{cluster_loss_.item()=}')

    return ae, pool, mean_loss


def cluster_loss(distances):
    min_dist = torch.min(distances, 1)
    return torch.mean(min_dist[0])


def test(inp: Tensor, labels, ae: AutoEncoder, pool: CentroidPool) -> pd.DataFrame:
    with torch.no_grad():
        latent, reconstruction = ae(inp)
        #cluster_distances = pool(latent)
        #predicted_clusters = torch.argmin(cluster_distances, 1)

        predicted_clusters = pool(latent)#assignments.max(-1)[1]
        label_idxs = np.argmax(pd.get_dummies(labels.values).values, axis=-1)

        data_cols = [f'dim_{i}' for i in range(latent.shape[1])]
        df = pd.DataFrame(latent.numpy(), columns=data_cols)
        df['predicted_cluster'] = predicted_clusters
        df['real_cluster'] = label_idxs

    return df


def eval_results(df: pd.DataFrame, pool: CentroidPool, inp):
    data_cols = [f'dim_{i}' for i in range(2)]
    projected = df[data_cols].values
    centroids = pool.coords.detach().numpy()
    pca = PCA(n_components=2)

    pca_transformed_rnaseq = pca.fit_transform(inp)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    sns.scatterplot(
        x=projected[:,0], 
        y=projected[:,1],
        hue=pd.Categorical(df['real_cluster']), 
        ax=ax[0],
        legend=False
    ).set_title('Real cell type')

    sns.scatterplot(
        x=projected[:,0], 
        y=projected[:,1],
        hue=pd.Categorical(df['predicted_cluster']), 
        ax=ax[1],
        legend=False
    ).set_title('Predicted cell type')

    sns.scatterplot(
        x=pca_transformed_rnaseq[:,0],
        y=pca_transformed_rnaseq[:,1],
        hue=pd.Categorical(df['real_cluster']), 
        ax=ax[2],
        legend=False
    ).set_title('PCA Transformed Data')

    ax[1].scatter(centroids[:,0], centroids[:,1], marker='*', color='black')

    plt.show()


def gather_clusters():
    dataset, inputs, labels = dataprocessing()
    ae = AutoEncoder([1200, 256, 128, 2])
    # pool = CentroidPool(11, 2)
    pool = KMadness(11, 2)

    ae, pool, loss = train(dataset, ae, pool, 1)
    df = test(inputs, labels, ae, pool)
    #eval_results(df, pool, inputs)


if __name__ == "__main__":
    gather_clusters()
