import torch.nn as nn

from src.models.hgt import HGTEncoder
from src.models.decoder import DistMultDecoder

class BioGraphModel(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.encoder = HGTEncoder(metadata, hidden_dim, out_dim, num_heads, num_layers)
        self.decoder = DistMultDecoder(out_dim)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index):
        drug_emb = z_dict["drug"][edge_label_index[0]]
        disease_emb = z_dict["disease"][edge_label_index[1]]
        return self.decoder(drug_emb, disease_emb)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(z_dict, edge_label_index)