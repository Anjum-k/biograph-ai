import pandas as pd

# Hetionet uses its own node-kind and relation names; our downstream code
# (split.py's TARGET_EDGE, model configs, dashboard) expects the simpler
# 'drug' / 'disease' / 'gene' / 'treats' / 'targets' / 'associated_with'
# vocabulary. Normalize once, right after loading, so everything downstream
# stays consistent regardless of which raw dataset we loaded from.
"""create_id_mappings का पूरा explanation
यह function क्यों चाहिए

Hetionet में हर node का ID एक text होता है,
जैसे Compound::DB00001 या Gene::1। PyTorch ऐसे text को नहीं समझता,
उसे सिर्फ़ numbers चाहिए। यह function हर text ID को एक छोटा integer दे देता है।
        """

KIND_RENAME = {
    "Compound": "drug",
    "Disease": "disease",
    "Gene": "gene",
}

RELATION_RENAME = {
    "CtD": "treats",
    "CbG": "targets",
    "DaG": "associated_with",
}


def normalize_names(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Rename Hetionet's 'kind'/'relation' values to our standard vocabulary."""
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()

    nodes_df["kind"] = nodes_df["kind"].replace(KIND_RENAME)

    if "relation" not in edges_df.columns and "metaedge" in edges_df.columns:
        edges_df = edges_df.rename(columns={"metaedge": "relation"})
    edges_df["relation"] = edges_df["relation"].replace(RELATION_RENAME)

    return nodes_df, edges_df


def create_id_mappings(nodes_df: pd.DataFrame) -> dict:
    """Return {node_type: {original_id: integer_index}}."""
    mappings = {}
    for node_type in nodes_df["kind"].unique():
        subset = nodes_df[nodes_df["kind"] == node_type].reset_index(drop=True)
        mappings[node_type] = {row["id"]: idx for idx, row in subset.iterrows()}
    return mappings


def map_edges_to_indices(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Attach source/target kind + integer index columns to the edge table."""
    id_to_kind = dict(zip(nodes_df["id"], nodes_df["kind"]))

    edges_df = edges_df.copy()
    # Hetionet's edge file names the relation column 'metaedge', not 'relation' —
    # normalize it here so the rest of the pipeline can always use 'relation'.
    if "relation" not in edges_df.columns and "metaedge" in edges_df.columns:
        edges_df = edges_df.rename(columns={"metaedge": "relation"})
    edges_df["source_kind"] = edges_df["source"].map(id_to_kind)
    edges_df["target_kind"] = edges_df["target"].map(id_to_kind)
    edges_df = edges_df.dropna(subset=["source_kind", "target_kind"])

    # Vectorized mapping (fast) instead of row-wise .apply() (which can take
    # hours on Hetionet's ~2.2 million edges). We build one flat {id: idx}
    # lookup per node kind and use pandas' vectorized .map().
    source_idx = pd.Series(index=edges_df.index, dtype="Int64")
    target_idx = pd.Series(index=edges_df.index, dtype="Int64")

    for kind, mapping in mappings.items():
        src_mask = edges_df["source_kind"] == kind
        source_idx[src_mask] = edges_df.loc[src_mask, "source"].map(mapping)

        tgt_mask = edges_df["target_kind"] == kind
        target_idx[tgt_mask] = edges_df.loc[tgt_mask, "target"].map(mapping)

    edges_df["source_idx"] = source_idx
    edges_df["target_idx"] = target_idx
    edges_df = edges_df.dropna(subset=["source_idx", "target_idx"])
    edges_df["source_idx"] = edges_df["source_idx"].astype(int)
    edges_df["target_idx"] = edges_df["target_idx"].astype(int)
    return edges_df
