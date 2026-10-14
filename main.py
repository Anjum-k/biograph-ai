"""CLI entry point: train, evaluate, or explain."""
import argparse

import pandas as pd
import torch

from src.data.id_mapping import create_id_mappings, map_edges_to_indices, normalize_names
from src.data.build_graph import build_hetero_graph
from src.data.split import split_drug_disease_edges
from src.training.train import train_model
from src.training.evaluate import full_evaluation
from src.utils.config import DATA_RAW, EXPERIMENTS_DIR, MODEL_CONFIG, TRAIN_CONFIG


def load_and_build_graph(use_subset: bool = False):
    from src.utils.config import DATA_PROCESSED

    if use_subset:
        print("Loading SUBSET data (fast test mode)...")
        nodes_df = pd.read_csv(DATA_PROCESSED / "subset_nodes.tsv", sep="\t")
        edges_df = pd.read_csv(DATA_PROCESSED / "subset_edges.tsv", sep="\t")
    else:
        print("Loading FULL Hetionet data (this can take a while on the mapping step)...")
        nodes_df = pd.read_csv(DATA_RAW / "hetionet_nodes.tsv", sep="\t")
        edges_df = pd.read_csv(DATA_RAW / "hetionet_edges.tsv", sep="\t")

    print(f"  {len(nodes_df)} nodes, {len(edges_df)} edges loaded")

    nodes_df, edges_df = normalize_names(nodes_df, edges_df)

    print("Building ID mappings...")
    mappings = create_id_mappings(nodes_df)

    print("Mapping edges to indices (vectorized)...")
    edges_df = map_edges_to_indices(edges_df, nodes_df, mappings)

    print("Building HeteroData graph object...")
    data = build_hetero_graph(edges_df, mappings)
    print("Graph built:", data)
    return data, mappings


def run_train(use_subset: bool = False):
    data, _ = load_and_build_graph(use_subset=use_subset)
    train_data, val_data, test_data = split_drug_disease_edges(data)

    model = train_model(
        train_data, val_data, metadata=data.metadata(),
        num_epochs=TRAIN_CONFIG["num_epochs"], lr=TRAIN_CONFIG["lr"],
        hidden_channels=MODEL_CONFIG["hidden_channels"],
        out_channels=MODEL_CONFIG["out_channels"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
    )

    test_results = full_evaluation(model, test_data)
    print("Test Results:", test_results)


def run_explain(drug: str, disease: str):
    print(
        f"Explain मोड को अपने graph_data + name->index मैपिंग के साथ जोड़ें "
        f"(देखें src/explainability/) — यहाँ {drug} -> {disease} के लिए स्क्रिप्ट attach करें।"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "explain"], required=True)
    parser.add_argument("--drug", type=str, default=None)
    parser.add_argument("--disease", type=str, default=None)
    parser.add_argument("--subset", action="store_true",
                         help="Use the small pre-built subset in data/processed/ instead of the full graph")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(use_subset=args.subset)
    elif args.mode == "explain":
        run_explain(args.drug, args.disease)
