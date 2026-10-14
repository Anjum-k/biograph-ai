import pandas as pd
from torch import nn

from src.data.download import RAW_DIR
from src.data.id_mapping import normalize_names, create_id_mappings, map_edges_to_indices
from src.data.build_graph import build_hetero_graph
from src.data.split import split_drug_disease_edges, TARGET_EDGE
from src.models.biograph_model import BioGraphModel

nodes = pd.read_csv(RAW_DIR / "hetionet_nodes.tsv", sep="\t")
edges = pd.read_csv(RAW_DIR / "hetionet_edges.tsv", sep="\t")

nodes, edges = normalize_names(nodes, edges)
mappings = create_id_mappings(nodes)
edges = map_edges_to_indices(edges, nodes, mappings)

data = build_hetero_graph(edges, mappings)
train_data, val_data, test_data = split_drug_disease_edges(data)

model = BioGraphModel(train_data.metadata())

label_index = train_data[TARGET_EDGE].edge_label_index
labels = train_data[TARGET_EDGE].edge_label

scores = model(train_data.x_dict, train_data.edge_index_dict, label_index)
loss = nn.BCEWithLogitsLoss()(scores, labels)

print("scores shape:", tuple(scores.shape))
print("loss:", round(loss.item(), 4))