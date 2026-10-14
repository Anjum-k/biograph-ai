"""Extract and rank the neighbors that most influenced a node's embedding,
using the attention weights returned by HGTConv layers."""
import torch


def extract_top_attended_neighbors(attn_weights, node_id, node_type, top_k=5):
    """
    attn_weights: list of {relation: (edge_index, weights)} per layer,
                  as returned by HGTConv(..., return_attention_weights=True)
    """
    results = {}
    for relation, (edge_idx, weights) in attn_weights.items():
        src_type, rel_name, dst_type = relation
        if dst_type != node_type:
            continue

        mask = edge_idx[1] == node_id
        relevant_edges = edge_idx[:, mask]
        relevant_weights = weights[mask]
        if relevant_weights.numel() == 0:
            continue

        top_vals, top_idx = torch.topk(
            relevant_weights.mean(dim=-1), k=min(top_k, relevant_weights.size(0))
        )
        top_sources = relevant_edges[0][top_idx]
        results[relation] = list(zip(top_sources.tolist(), top_vals.tolist()))

    return results
