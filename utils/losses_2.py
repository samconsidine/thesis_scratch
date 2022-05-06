
import scanpy as sc
from torch import Tensor
from torch.nn import MSELoss, Module
from dataclasses import dataclass


def load_models():
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
    return encoder, decoder

    latent_dim = 2
    data = sc.datasets.paul15()
    X = torch.tensor(data.X).float()

    
    centroids = Centroids(4, 2)
    optimizer = torch.optim.Adam(list(centroids.parameters()) + list(encoder.parameters()) + list(decoder.parameters()))

    n_epochs = 100
    for epoch in range(n_epochs):

        latent = encoder(X)
        mst = torch.tensor([[0, 1, 2], [1, 2, 3]])  # This

        midpoints = (centroids.coords[mst[0]] + centroids.coords[mst[1]]) / 2  # This (less so)

        distances = torch.cdist(latent, midpoints, p=2.0)
        projection_probabilities = torch.nn.functional.softmax(distances)
        #projected_coords = project_onto_mst(x, midpoints)
        projected_coords = midpoints.unsqueeze(1).repeat(1, X.shape[0], 1)   # Maybe this will change?

        reconstructions = [decoder(mst_projection) for mst_projection in projected_coords]
        prediction_target_pairs = zip(projection_probabilities, reconstructions)

        loss = sum(
            (projection_probabilities[:, i].unsqueeze(0).T 
                * (torch.abs(X - reconstruction)**2).sqrt().sum(1)).mean()
            for i, reconstruction in enumerate(reconstructions)
        )
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


