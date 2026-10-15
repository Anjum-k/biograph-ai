from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


from src.data.download import RAW_DIR
from src.data.id_mapping import normalize_names, create_id_mappings, map_edges_to_indices
from src.data.build_graph import build_hetero_graph
from src.data.split import split_drug_disease_edges, TARGET_EDGE
from src.models.biograph_model import BioGraphModel

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "experiments" / "best_model.pt"


def load_data():
    nodes = pd.read_csv(RAW_DIR / "hetionet_nodes.tsv", sep="\t")
    edges = pd.read_csv(RAW_DIR / "hetionet_edges.tsv", sep="\t")

    nodes, edges = normalize_names(nodes, edges)
    mappings = create_id_mappings(nodes)
    edges = map_edges_to_indices(edges, nodes, mappings)

    data = build_hetero_graph(edges, mappings)
    return split_drug_disease_edges(data)

def roc_auc(labels, scores):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def avg_precision(labels, scores):
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return (precision * y).sum() / y.sum()

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    label_index = data[TARGET_EDGE].edge_label_index
    labels = data[TARGET_EDGE].edge_label

    scores = model(data.x_dict, data.edge_index_dict, label_index)
    probs = torch.sigmoid(scores).cpu().numpy()
    labels = labels.cpu().numpy()

    auroc = roc_auc(labels, probs)
    auprc = avg_precision(labels, probs)
    return auroc, auprc


def train_one_epoch(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    label_index = data[TARGET_EDGE].edge_label_index
    labels = data[TARGET_EDGE].edge_label

    scores = model(data.x_dict, data.edge_index_dict, label_index)
    loss = loss_fn(scores, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


def main(epochs=100, lr=0.001):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data, test_data = load_data()
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = BioGraphModel(train_data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_auroc = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_data, optimizer, loss_fn)
        val_auroc, val_auprc = evaluate(model, val_data)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d} | loss {loss:.4f} | val AUROC {val_auroc:.4f} | val AUPRC {val_auprc:.4f}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    test_auroc, test_auprc = evaluate(model, test_data)
    print(f"best val AUROC {best_val_auroc:.4f}")
    print(f"TEST  AUROC {test_auroc:.4f} | AUPRC {test_auprc:.4f}")


if __name__ == "__main__":
    main()