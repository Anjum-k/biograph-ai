"""Heterogeneous Graph Transformer encoder."""
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear



class HGTEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=64, num_heads=4,
                 num_layers=2, num_nodes_dict=None):
        super().__init__()
        node_types = metadata[0]
        self.use_embeddings = num_nodes_dict is not None

        if self.use_embeddings:
            self.embeddings = nn.ModuleDict(
                {t: nn.Embedding(num_nodes_dict[t], hidden_dim) for t in node_types}
            )
        else:
            self.input_proj = nn.ModuleDict(
                {t: Linear(-1, hidden_dim) for t in node_types}
            )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, num_heads) for _ in range(num_layers)]
        )

        self.out_proj = nn.ModuleDict(
            {node_type: Linear(hidden_dim, out_dim) for node_type in node_types}
        )

    def forward(self, x_dict, edge_index_dict):
        if self.use_embeddings:
            x_dict = {t: self.embeddings[t].weight for t in x_dict}
        else:
            x_dict = {
                node_type: self.input_proj[node_type](x).relu()
                for node_type, x in x_dict.items()
            }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return {
            node_type: self.out_proj[node_type](x)
            for node_type, x in x_dict.items()
        }