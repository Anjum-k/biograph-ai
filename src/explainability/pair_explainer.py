import pandas as pd
import torch

from src.data.download import RAW_DIR
from src.data.id_mapping import normalize_names, create_id_mappings
from src.data.split import TARGET_EDGE
from src.models.biograph_model import BioGraphModel
from src.training.train import load_data, CHECKPOINT_PATH


def load_names():
    nodes = pd.read_csv(RAW_DIR / "hetionet_nodes.tsv", sep="\t")
    edges = pd.read_csv(RAW_DIR / "hetionet_edges.tsv", sep="\t")
    nodes, edges = normalize_names(nodes, edges)
    mappings = create_id_mappings(nodes)

    id_to_name = dict(zip(nodes["id"], nodes["name"]))
    names = {}
    for kind, mapping in mappings.items():
        names[kind] = {idx: id_to_name[node_id] for node_id, idx in mapping.items()}
    return names


def neighbors(data, edge_type, src_idx):
    edge_index = data[edge_type].edge_index
    return set(edge_index[1][edge_index[0] == src_idx].tolist())


def shared_genes(data, drug_idx, disease_idx):
    drug_genes = neighbors(data, ("drug", "targets", "gene"), drug_idx)
    disease_genes = neighbors(data, ("disease", "associated_with", "gene"), disease_idx)
    return sorted(drug_genes & disease_genes)


def main(seed=42, top_n=10):
    torch.manual_seed(seed)
    names = load_names()
    train_data, val_data, test_data = load_data()

    model = BioGraphModel(test_data.metadata())
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    label_index = test_data[TARGET_EDGE].edge_label_index
    labels = test_data[TARGET_EDGE].edge_label
    with torch.no_grad():
        probs = torch.sigmoid(
            model(test_data.x_dict, test_data.edge_index_dict, label_index)
        )

    rows = []
    for i in range(label_index.size(1)):
        if labels[i].item() != 1:
            continue
        drug_idx = label_index[0, i].item()
        disease_idx = label_index[1, i].item()
        genes = shared_genes(test_data, drug_idx, disease_idx)
        rows.append((probs[i].item(), drug_idx, disease_idx, genes))

    rows.sort(reverse=True)
    for prob, drug_idx, disease_idx, genes in rows[:top_n]:
        gene_names = [names["gene"][g] for g in genes[:3]]
        print(
            f"{names['drug'][drug_idx]} -> {names['disease'][disease_idx]} "
            f"| prob {prob:.3f} | shared genes {len(genes)}: {gene_names}"
        )


if __name__ == "__main__":
    main()