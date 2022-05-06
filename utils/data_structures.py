from torch import Tensor
from dataclasses import dataclass


@dataclass
class MST:
    nodes: Tensor
    edges: Tensor
    probabilities: Tensor

    @property
    def midpoints(self):
        return (self.nodes[self.edges[0]] + self.nodes[self.edges[1]]) / 2

