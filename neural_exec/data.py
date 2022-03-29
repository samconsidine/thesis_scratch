import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected

from typing import Tuple, List

INF = torch.tensor(torch.inf, dtype=torch.float32)


def gen_prims_data_instance(n_nodes: int, n_dims: int) -> torch_geometric.data.Data:
    edge_weights, edge_index = gen_random_fc_graph(n_nodes, n_dims)
    mst = prims(edge_weights)
    data = Data(edge_attr=edge_weights, edge_index=edge_index, y=mst)
    return data


def gen_random_fc_graph(n_nodes:int, n_dims: int) -> Tuple[torch.Tensor, List[int]]:
    edge_index = [(x, y) for x in range(n_nodes) for y in range(n_nodes)]
    node_positions = torch.rand(n_nodes, n_dims)
    node_distances = torch.cdist(node_positions, node_positions)
    return node_distances, edge_index


def prims(edge_weights: torch.Tensor) -> torch.Tensor:
    dim = edge_weights.shape[0]
    visited = torch.zeros(dim, dtype=torch.bool)
    visited[0] = True
    mst = []

    while not torch.all(visited):
        next_node_flat_idx = torch.argmin(mask_visited(edge_weights, visited))
        to_node_idx = torch.remainder(next_node_flat_idx, dim)
        from_node_idx = torch.div(next_node_flat_idx, dim, rounding_mode='trunc')
        visited[to_node_idx] = True
        mst.append((from_node_idx.item(), to_node_idx.item()))

    mst = to_dense_adj(to_undirected(torch.tensor(mst).T))
    return mst


def mask_visited(edge_weights, visited: torch.Tensor) -> torch.Tensor:
    not_visited = torch.logical_not(visited)
    masked_weights = edge_weights * visited.unsqueeze(1) * not_visited
    return torch.where(masked_weights != 0, masked_weights, INF)

