from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from dataclasses import dataclass
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProcessorNetwork(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='max', bias=False,  # Channels?
            flow='source_to_target', use_gru=False):

        super(ProcessorNetwork, self).__init__(aggr=aggr, flow=flow)

        self.M = nn.Sequential(
            nn.Linear(2*in_channels+1, out_channels, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels, bias=bias),
            nn.LeakyReLU()
        )

        self.U = nn.Sequential(
            nn.Linear(2*in_channels, out_channels, bias=bias),
            nn.LeakyReLU()
        )

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRUCell(out_channels, out_channels, bias=bias)

        self.out_channels = out_channels

        self.to(device)

    def forward(self, x, edge_attr, edge_index, hidden):
        out = self.propagate(edge_index, x=x, hidden=hidden, edge_attr=edge_attr)

        if not self.training:
            out = torch.clamp(out, -1e9, 1e9)

        return out

    def message(self, x_i, x_j, edge_attr):
        edge_weights_col_vec = edge_attr.unsqueeze(0).T
        return self.M(torch.cat((x_i, x_j, edge_weights_col_vec), dim=1))

    def update(self, aggr_out, x, hidden):

        if self.use_gru:
            out = self.gru(self.U(torch.cat((x, aggr_out), dim=1)), hidden)
        else:
            out = self.U(torch.cat((x, aggr_out), dim=1))

        return out


class Encoder(nn.Module):
    def __init__(self, node_feature_dim: int, latent_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(node_feature_dim + latent_dim, latent_dim),
            nn.ReLU()
        )

        self.to(device)

    def forward(self, prev_tree: Tensor, latent: Tensor) -> Tensor:
        model_in = torch.cat([prev_tree, latent], axis=1)
        return self.layers(model_in)


class MSTDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim * 2, 1),
            nn.Sigmoid()
        )

        self.to(device)

    def forward(self, encoded: Tensor, h: Tensor) -> Tensor:
        model_in = torch.cat([encoded, h], axis=1)
        return self.layers(model_in)


class PredecessorDecoder(nn.Module):
    def __init__(self, latent_dim: int, n_outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        self.n_outputs = n_outputs
        self.to(device)

    def forward(self, encoded: Tensor, h: Tensor, edge_index: Tensor) -> Tensor:
        left_encoded = encoded[edge_index[0]]
        right_encoded = encoded[edge_index[1]]
        left_h = h[edge_index[0]]
        right_h = h[edge_index[1]]

        out = self.layers(torch.cat(
            (left_encoded, left_h, right_encoded, right_h), axis=1))

        out = out.reshape((-1, self.n_outputs))
        return out


