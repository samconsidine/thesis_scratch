import torch

from scrnaseq_processor.data import pipeline
from scrnaseq_processor.models import AutoEncoder, KMadness, CentroidPool
from scrnaseq_processor.main import train, test, eval_results
from neural_exec.create_prims_model import (
    create_prims_model, create_tree, ProcessorNetwork
)
from utils.losses import mst_reconstruction_loss
from utils.data_structures import MST

import scanpy as sc
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def gather_clusters():
    dataset, inputs, labels = pipeline()
    ae = AutoEncoder([1200, 256, 2])
    pool = CentroidPool(11, 2)

    ae, pool, loss = train(dataset, ae, pool, 10)
    df = test(inputs, labels, ae, pool)
    # eval_results(df, pool, inputs) # Am going to need to redo how this works fml

    return dataset, ae, pool, inputs, labels


def solve_mst(prims_solver, cluster_centers):
    enc = prims_solver.encoder(cluster_centers)
    solved = prims_solver.processor(enc)
    dec = prims_solver.mst_decoder(solved)
    return dec


def plot_mst(logits, X, centroids):
    to_nodes = logits.argmax(1).detach().numpy()
    from_nodes = np.arange(len(to_nodes))
    centroids = centroids.detach().numpy()
    for i in range(len(centroids)):
        xs = [centroids[from_nodes[i]][0], centroids[to_nodes[i]][0]]
        ys = [centroids[from_nodes[i]][1], centroids[to_nodes[i]][1]]
        plt.plot(xs, ys, 'ro-')
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def join_parameters(*modules):
    return sum(map(lambda x: list(x.parameters()), modules), start=[])


def grad_of(module):
    if type(module) == torch.nn.Sequential:
        return module[0].weight.grad
    elif type(module) == ProcessorNetwork:
        return module.M[0].weight.grad
    elif type(module) == CentroidPool:
        return module.coords.grad
    else:
        return module.layers[0].weight.grad


def is_nonzero_grad(module):
    return (torch.count_nonzero(grad_of(module)) > 0).item()


def ensure_gradients(gene_encoder, gene_decoder, pool, mst_encoder, processor,
        mst_decoder, predecessor_decoder):
    print(f'{is_nonzero_grad(gene_encoder)=},\n'
          f'{is_nonzero_grad(gene_decoder)=},\n'
          f'{is_nonzero_grad(pool)=},\n'
          f'{is_nonzero_grad(mst_encoder)=},\n'
          f'{is_nonzero_grad(processor)=},\n'
          f'{is_nonzero_grad(mst_decoder)=},\n'
          f'{is_nonzero_grad(predecessor_decoder)=}')


if __name__ == "__main__":
    orig_data = sc.datasets.paul15()
    data, ae, pool, inputs, labels = gather_clusters()
    prims_solver = create_prims_model(num_nodes=pool.n_clusts)

    cluster_centers = pool.coords
    tree = prims_solver(cluster_centers)

    X = torch.cat([d[0] for d in data])
    #plot_mst(tree, ae.encoder(X).detach().numpy(), cluster_centers)

    optimizer = torch.optim.Adam(join_parameters(ae, pool, prims_solver))

    n_epochs = 10
    torch.set_printoptions(precision=9)
    for epoch in range(n_epochs):
        latent = ae.encoder(X)
        cluster_centers = pool.coords
        graph_size = cluster_centers.shape[0]
        tree_logits = prims_solver(cluster_centers)

        tree = torch.ones_like(tree_logits).nonzero().T
        mst = MST(cluster_centers, tree, tree_logits.softmax(1))
        loss = mst_reconstruction_loss(latent[:5000], mst, X[:5000], ae.decoder)

        optimizer.zero_grad()
        loss.backward()
        ensure_gradients(ae.encoder, ae.decoder, pool, prims_solver.encoder,
                prims_solver.processor, prims_solver.mst_decoder,
                prims_solver.predecessor_decoder)
        optimizer.step()

        with torch.no_grad():
            import seaborn as sns
            import matplotlib.pyplot as plt

            outs = ae.encoder(inputs).detach().numpy()
            xs = outs[:, 0]
            ys = outs[:, 1]

            coords = pool.coords
            cx = coords[:, 0]
            cy = coords[:, 1]

            sns.scatterplot(x=xs, y=ys, hue=labels.values)
            sns.scatterplot(x=cx, y=cy, marker="*", zorder=10, color='black')
            plt.show()

            from_nodes = mst.edges[0]
            to_nodes = mst.edges[1]
            # for i in range(len(coords)):
            #     xs = [coords[from_nodes[i]][0], coords[to_nodes[i]][0]]
            #     ys = [coords[from_nodes[i]][1], coords[to_nodes[i]][1]]
            #     plt.plot(xs, ys, 'ro-')

    plot_mst(tree_logits, ae.encoder(X).detach().numpy(), cluster_centers)

