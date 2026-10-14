from src.data.download import RAW_DIR
import pandas as pd

from src.data.id_mapping import normalize_names, create_id_mappings, map_edges_to_indices
from src.data.build_graph import build_hetero_graph
from src.data.split import split_drug_disease_edges, TARGET_EDGE

nodes = pd.read_csv(RAW_DIR / "hetionet_nodes.tsv", sep="\t")
edges = pd.read_csv(RAW_DIR / "hetionet_edges.tsv", sep="\t")

nodes, edges = normalize_names(nodes, edges)
mappings = create_id_mappings(nodes)
edges = map_edges_to_indices(edges, nodes, mappings)

data = build_hetero_graph(edges, mappings)
train_data, val_data, test_data = split_drug_disease_edges(data)

for name, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
    label = d[TARGET_EDGE].edge_label
    print(f"{name}: positive={int(label.sum())}, negative={int((label == 0).sum())}")

print("Edge types:", len(train_data.edge_types))