import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_undirected
import matplotlib.pyplot as plt

from typing import Tuple, List, Iterable


INF = torch.tensor(torch.inf, dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gen_prims_data_instance(n_nodes: int, n_dims: int) -> Data:
    edge_weights, edge_index = gen_random_fc_graph(n_nodes, n_dims)

    generator = prims_generator(n_nodes, edge_weights)
    prims_steps = [item for item in prims_generator(n_nodes, edge_weights)]

    x_next, x_prev, predecessor = (torch.stack(x, axis=1) for x in zip(*prims_steps))
    predecessor = predecessor.T  # To get correct shape for batching

    return Data(
        x=x_prev.to(device), 
        y=x_next.to(device), 
        edge_weights=flatten_edge_weights(edge_weights, edge_index).to(device),
        edge_index=edge_index.to(device), 
        predecessor=predecessor.to(device),
        graph_size=n_nodes
    )


def flatten_edge_weights(edge_weights: Tensor, edge_index: Tensor) -> Tensor:

    def test_flattened_weights(edge_weights: Tensor, flat_weights: Tensor):
        for i in range(flat_weights.shape[0]):
            assert edge_weights[edge_index[0][i], edge_index[1][i]] == flat_weights[i]

    flat_weights = edge_weights.flatten()

    test_flattened_weights(edge_weights, flat_weights)
    return flat_weights


def print_instance(data: Data):
    print(f"predecessors: {data.predecessor_index.T}")
    print(f"prev tree: {data.x}")
    print(f"next tree: {data.y}")
    print(f"edge_weights: {data.edge_weights}")


def gen_random_fc_graph(n_nodes:int, n_dims: int) -> Tuple[Tensor, List[int]]:
    edge_index = [(x, y) for x in range(n_nodes) for y in range(n_nodes)]
    node_positions = torch.rand(n_nodes, n_dims)
    node_distances = torch.cdist(node_positions, node_positions)
    return node_distances, torch.tensor(edge_index).T


def prims_generator(num_nodes: int, edge_weights: Tensor) -> Tensor:
    dim = edge_weights.shape[0]
    visited = torch.zeros(dim, dtype=torch.bool)
    visited[0] = True
    predecessor = torch.zeros(num_nodes)
    predecessor.fill_(torch.nan)
    predecessor[0] = 0.

    for _ in range(num_nodes-1):
        prev_visited = visited.clone()

        next_node_flat_idx = torch.argmin(mask_visited(edge_weights, visited)).item()
        from_node_idx, to_node_idx = divmod(next_node_flat_idx, dim) 
        visited[to_node_idx] = True
        predecessor[to_node_idx] = from_node_idx

        yield visited.float(), prev_visited.float(), predecessor.clone()


def mask_visited(edge_weights, visited: Tensor) -> Tensor:
    not_visited = torch.logical_not(visited)
    masked_weights = edge_weights * visited.unsqueeze(1) * not_visited
    return torch.where(masked_weights != 0, masked_weights, INF)


def generate_prims_dataset(size: int, num_nodes: int, batch_size: int) -> DataLoader:
    graphs = [gen_prims_data_instance(num_nodes, 1) for _ in range(size)]
    loader = DataLoader(graphs, batch_size=batch_size)
    return loader


if __name__=="__main__":
    graphs = [gen_prims_data_instance(4, 1) for _ in range(2)]
    print(graphs[0].edge_weights)
    print(graphs[1].edge_weights)
    loader = DataLoader(graphs, batch_size=2)
    for data in loader:
        print_instance(data)

