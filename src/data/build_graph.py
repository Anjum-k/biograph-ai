"""Build a PyTorch Geometric HeteroData graph from mapped edges."""
import torch
from torch_geometric.data import HeteroData

KEEP_KINDS = ["drug", "gene", "disease"]
KEEP_RELATIONS = ["treats", "targets", "associated_with"]


def build_hetero_graph(edges_df, mappings, feature_dim: int = 64) -> HeteroData:
    data = HeteroData()

    for node_type in KEEP_KINDS:
        n = len(mappings[node_type])
        data[node_type].num_nodes = n
        data[node_type].x = torch.randn(n, feature_dim)

    mask = (
        edges_df["source_kind"].isin(KEEP_KINDS)
        & edges_df["target_kind"].isin(KEEP_KINDS)
        & edges_df["relation"].isin(KEEP_RELATIONS)
    )
    edges_df = edges_df[mask]

    for (src_type, relation, tgt_type), group in edges_df.groupby(
        ["source_kind", "relation", "target_kind"]
    ):
        edge_index = torch.tensor(
            group[["source_idx", "target_idx"]].values.T, dtype=torch.long
        )
        data[src_type, relation, tgt_type].edge_index = edge_index

    return data