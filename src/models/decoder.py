import torch
import torch.nn as nn


class DistMultDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relation = nn.Parameter(torch.ones(dim))

    def forward(self, drug_emb, disease_emb):
        return (drug_emb * self.relation * disease_emb).sum(dim=-1)