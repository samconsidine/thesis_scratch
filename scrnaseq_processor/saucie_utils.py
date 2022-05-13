import torch

def get_cluster_merging(clusters, embeddings, merge_k_nearest):
    """
    Merge clusters based on bi-directional best hit using MMD distance between clusters.
    """
    if len(np.unique(clusters))==1: return clusters
    clusts_to_use = np.unique(clusters)
    mmdclusts = np.zeros((len(clusts_to_use), len(clusts_to_use)))
    for i1, i2 in itertools.combinations(range(len(clusts_to_use)), 2):
        clust1 = clusts_to_use[i1]
        clust2 = clusts_to_use[i2]
        if clust1 == -1 or clust2 == -1:
            continue
        ei = embeddings[clusters == clust1]
        ej = embeddings[clusters == clust2]
        ri = list(range(ei.shape[0])); np.random.shuffle(ri); ri = ri[:1000];
        rj = list(range(ej.shape[0])); np.random.shuffle(rj); rj = rj[:1000];
        ei = ei[ri, :]
        ej = ej[rj, :]
        k1 = sklearn.metrics.pairwise.pairwise_distances(ei, ei)
        k2 = sklearn.metrics.pairwise.pairwise_distances(ej, ej)
        k12 = sklearn.metrics.pairwise.pairwise_distances(ei, ej)
        mmd = 0
        for sigma in [.01, .1, 1., 10.]:
            k1_ = np.exp(- k1 / (sigma**2))
            k2_ = np.exp(- k2 / (sigma**2))
            k12_ = np.exp(- k12 / (sigma**2))
            mmd += k1_.sum()/(k1_.shape[0]*k1_.shape[1]) +\
                   k2_.sum()/(k2_.shape[0]*k2_.shape[1]) -\
                   2*k12_.sum()/(k12_.shape[0]*k12_.shape[1])
        mmdclusts[i1, i2] = mmd
        mmdclusts[i2, i1] = mmd

    clust_reassign = {}
    for i1, i2 in itertools.combinations(range(mmdclusts.shape[0]), 2):
        k5_1 = np.argsort(mmdclusts[i1, :])[1:merge_k_nearest+1]
        k5_2 = np.argsort(mmdclusts[i2, :])[1:merge_k_nearest+1]
        if np.isin(i2, k5_1) and np.isin(i1, k5_2):
            clust_reassign[clusts_to_use[i2]] = clusts_to_use[i1]
            clusts_to_use[i2] = clusts_to_use[i1]

    for c in clust_reassign:
        mask = clusters == c
        clusters[mask] = clust_reassign[c]
    return clusters

def get_clusters(acts, embedding_DFs, binmin=10, max_clusters=100, merge_k_nearest=1):
    """
    Clustering function. Binarize activations on clustering layer and aggregate binary codes.
    """
    acts = acts / acts.max()
    binarized = np.where(acts > 1e-6, 1, 0)
    unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
    unique_rows = unique_rows[counts > binmin]
    n_clusters = unique_rows.shape[0]

    n_clusters = 0
    clusters = -1 * np.ones(acts.shape[0])
    for i, row in enumerate(unique_rows):
        rows_equal_to_this_code = np.where(np.all(binarized == row, axis=1))[0]

        clusters[rows_equal_to_this_code] = n_clusters
        n_clusters += 1

    print(f'{len(np.unique(clusters))} clusters detected. Merging clusters...')
    print(f'\t{(clusters == -1).astype(int).sum()} cells '
        f'({100. * (clusters == -1).astype(int).sum() / float(len(clusters)):.2f} % of total) are not clustered.')
    # merge clusters
    embeddings = np.vstack([df.values for df in embedding_DFs])
    clusters = get_cluster_merging(clusters, embeddings, merge_k_nearest)
    n_clusters = len(np.unique(clusters))
    print(f'Merging done. Total {len(np.unique(clusters))} clusters.')
    return n_clusters, clusters

