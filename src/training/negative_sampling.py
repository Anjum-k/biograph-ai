"""Negative sampling for drug-disease link prediction (used at inference-time
re-sampling; RandomLinkSplit already handles it for the static train split)."""
import torch


def sample_negative_edges(num_drugs: int, num_diseases: int, positive_edge_index: torch.Tensor,
                           num_neg_samples: int) -> torch.Tensor:
    pos_set = set(zip(positive_edge_index[0].tolist(), positive_edge_index[1].tolist()))
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        d = torch.randint(0, num_drugs, (1,)).item()
        dis = torch.randint(0, num_diseases, (1,)).item()
        if (d, dis) not in pos_set:
            neg_edges.append((d, dis))
    neg_edges = torch.tensor(neg_edges, dtype=torch.long).T
    return neg_edges
