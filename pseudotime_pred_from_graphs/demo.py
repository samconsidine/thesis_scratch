from model import KNNGraphInferenceModel
from scvi import data
import scanpy as sc


def load_model():
    model = KNNGraphInferenceModel(input_dim=1200, output_dim=3)
    return model


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

def load_to_batches(data):
    inp = torch.tensor(data.X.toarray())
    labels = data.obs['cell_type']
    ae = AutoEncoder([data.n_vars, 256, 2])

    dataset = [(inp[i:i+128], torch.tensor(pd.get_dummies(labels[i:i+128]).values)) for i in range(0, len(inp), 128)]
    return dataset


if __name__ == "__main__":
    data = load_data()
    model = load_model()

    model = train_model(model, data)

