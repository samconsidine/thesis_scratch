from torch_geometric.data import Data
from graph_utils import increase_self_loop_weight


def preprocess_mst(data: Data) -> Data:
    data.edge_weights = increase_self_loop_weight(data.edge_weights, data.edge_index, 1.)
    return data

