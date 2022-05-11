import torch 
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from dataclasses import dataclass
from torch_geometric.data import Data

from neural_exec.model import ProcessorNetwork, Encoder, MSTDecoder, PredecessorDecoder
from neural_exec.preprocessing import preprocess_mst
from neural_exec.prims import generate_prims_dataset


def batch_mst_acc(preds, real):
    with torch.no_grad():
        return (preds == real).float().mean()

        breakpoint()

class PrimsSolver(torch.nn.Module):
    def __init__(
        self, 
        encoder: Encoder,
        processor: ProcessorNetwork,
        mst_decoder: MSTDecoder,
        predecessor_decoder: PredecessorDecoder
    ):
        super().__init__()

        self.encoder = encoder
        self.processor = processor
        self.mst_decoder = mst_decoder
        self.predecessor_decoder = predecessor_decoder
        self.num_nodes = 11

    def forward(self, data: Tensor) -> Tensor:
        latent_dim = 16
        h = torch.zeros((self.num_nodes, latent_dim))
        prev_tree = torch.zeros(self.num_nodes, 1).long()
        edge_index = torch.tensor([[x, y] for x in range(self.num_nodes) 
                                   for y in range(self.num_nodes)]).T.long()
        edge_weights = torch.cdist(data, data).flatten()

        for step in range(self.num_nodes - 1):
            encoded = self.encoder(prev_tree, h)
            h = self.processor(x=encoded, edge_attr=edge_weights,
                               edge_index=edge_index, hidden=h)

            mst_logits = self.mst_decoder(encoded, h)
            pred_logits = self.predecessor_decoder(encoded, h, edge_index)
            prev_tree[mst_logits.argmax()] = 1

        return pred_logits


def create_prims_model(latent_dim=16, node_features=1, num_nodes=6, mst_coef=1.,
        pred_coef=1., n_data=2, batch_size=1, epochs=5000):

    loader = generate_prims_dataset(size=n_data, num_nodes=num_nodes,
                                    batch_size=batch_size)
    encoder = Encoder(node_feature_dim=1, latent_dim=latent_dim)
    processor = ProcessorNetwork(in_channels=latent_dim,
                                 out_channels=latent_dim, bias=True)
    mst_decoder = MSTDecoder(latent_dim)
    pred_decoder = PredecessorDecoder(latent_dim, num_nodes * batch_size)

    mst_loss_fn = torch.nn.BCELoss()
    pred_loss_fn = torch.nn.CrossEntropyLoss()

    flatten = lambda x: sum(x, start=[])

    models = [encoder, processor, mst_decoder, pred_decoder]
    params_list = flatten(list(x.parameters()) for x in models)
    optimizer = torch.optim.Adam(params_list)

    losses = []
    for epoch in tqdm(range(epochs)):
        batch_ctr = 0.
        acc_avg = 0.

        # Train
        for data in loader:
            graph_size = data.graph_size[0]
            h = torch.zeros((data.num_nodes, latent_dim))
            mst_loss = 0.
            pred_loss = 0.

            for step in range(graph_size - 1):
                prev_tree = data.x[:, step:(step+1)]  # Keep dims of slice
                current_tree = data.y[:, step:(step+1)]
                predecessors = data.predecessor[-1].long()

                encoded = encoder(prev_tree, h)
                h = processor(x=encoded, edge_attr=data.edge_weights,
                              edge_index=data.edge_index, hidden=h)

                mst_logits = mst_decoder(encoded, h)
                pred_logits = pred_decoder(encoded, h, data.edge_index)

                mst_loss += mst_loss_fn(mst_logits, current_tree)
                pred_loss += pred_loss_fn(pred_logits, predecessors)

                in_mst = (mst_logits > 0.5).float()  # Why > 0?
                acc_avg += batch_mst_acc(in_mst, current_tree)
                batch_ctr += 1

            # loss = MST_COEF * mst_loss + PRED_COEF * pred_loss
            loss = pred_loss + mst_loss
            loss /= data.graph_size[0] - 1
            # print(f":Loss: {loss.item()}")
            losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"{pred_logits=}")
    print(f"{predecessors=}")

    return PrimsSolver(encoder, processor, mst_decoder, pred_decoder)


def create_tree(
        prims_solver:PrimsSolver, 
        cluster_centers:
        Tensor, graph_size: int
    ) -> Tensor:

    h = torch.zeros((graph_size, 16))  # latent dim
    prev_tree = torch.zeros((graph_size,1))
    prev_tree[0,0] = 1.

    predecessors = torch.zeros((graph_size,1))
    predecessors.fill_(torch.nan)
    predecessors[0,0] = 0
    edge_index = torch.tensor([[x, y] 
        for x in range(graph_size) 
        for y in range(graph_size)]).T

    edge_weights = ((cluster_centers[edge_index[0]] - cluster_centers[edge_index[1]])**2).sum(1).sqrt()

    for step in range(graph_size - 1):

        encoded = prims_solver.encoder(prev_tree, h)
        h = prims_solver.processor(x=encoded, edge_attr=edge_weights,
                      edge_index=edge_index, hidden=h)

        mst_logits = prims_solver.mst_decoder(encoded, h)
        pred_logits = prims_solver.predecessor_decoder(encoded, h, edge_index)

        # Get the max of the logits and the preds to be the tree
        prev_tree[torch.argmax(mst_logits)] = 1  # ???

    tree = torch.stack([torch.arange(prev_tree.shape[0]), prev_tree.argmax(1)])
    return tree


if __name__ == "__main__":
    solver = create_prims_model()

