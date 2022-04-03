import torch
from torch import Tensor
from torch_scatter import gather_csr, scatter, segment_csr

from numbers import Number


def increase_self_loop_weight(edge_weights: Tensor, edge_index: Tensor, amount: Number) -> Tensor:

    if len(edge_weights.shape) == 1:
        return torch.where(edge_index[0] == edge_index[1], edge_weights + amount, edge_weights)

    elif len(edge_weights.shape) == 2:
        return edge_weights + torch.eye(edge_weights.shape) * amount

    else:
        raise ValueError(f"Can't increase self loop weight on tensor of size {edge_weights.shape}")


