import scanpy as sc
from scvi import data
import torch
from anndata import AnnData
import pandas as pd

from typing import Tuple


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


def process_for_model(adata: AnnData) -> Tuple[torch.Tensor, pd.Series]:
    inp = torch.tensor(adata.X.toarray())
    labels = adata.obs['cell_type']
    return inp, labels


def batch_data(inp, labels):
    dataset = [(inp[i:i+128], torch.tensor(pd.get_dummies(labels[i:i+128]).values)) for i in range(0, len(inp), 128)]
    return dataset
