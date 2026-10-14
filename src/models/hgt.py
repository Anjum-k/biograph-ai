"""Heterogeneous Graph Transformer encoder."""
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear



class HGTEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        node_types = metadata[0]

        self.input_proj = nn.ModuleDict(
            {node_type: Linear(-1, hidden_dim) for node_type in node_types}
        )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, num_heads) for _ in range(num_layers)]
        )

        self.out_proj = nn.ModuleDict(
            {node_type: Linear(hidden_dim, out_dim) for node_type in node_types}
        )
    
    def forward(self, x_dict, edge_index_dict):
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