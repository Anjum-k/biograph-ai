import torch

from src.explainability.pair_explainer import load_names, shared_genes
from src.models.biograph_model import BioGraphModel
from src.training.train import load_data, CHECKPOINT_PATH

FINAL_MODEL_PATH = CHECKPOINT_PATH.parent / "final_model.pt"

PAIRS = [
    ("Irbesartan", "hypertension"),
    ("Ramipril", "type 2 diabetes mellitus"),
    ("Epirubicin", "prostate cancer"),
]


def score_pair(model, data, drug_idx, disease_idx, edge_index_dict=None):
    if edge_index_dict is None:
        edge_index_dict = data.edge_index_dict
    label_index = torch.tensor([[drug_idx], [disease_idx]])
    with torch.no_grad():
        return torch.sigmoid(model(data.x_dict, edge_index_dict, label_index)).item()


def remove_edge(edge_index, src_idx, dst_idx):
    keep = ~((edge_index[0] == src_idx) & (edge_index[1] == dst_idx))
    return edge_index[:, keep]


def without_gene(data, drug_idx, disease_idx, gene_idx):
    edges = dict(data.edge_index_dict)
    for edge_type, src, dst in [
        (("drug", "targets", "gene"), drug_idx, gene_idx),
        (("gene", "rev_targets", "drug"), gene_idx, drug_idx),
        (("disease", "associated_with", "gene"), disease_idx, gene_idx),
        (("gene", "rev_associated_with", "disease"), gene_idx, disease_idx),
    ]:
        edges[edge_type] = remove_edge(edges[edge_type], src, dst)
    return edges


def explain_pair(model, data, names, drug_idx, disease_idx):
    base = score_pair(model, data, drug_idx, disease_idx)
    print(f"{names['drug'][drug_idx]} -> {names['disease'][disease_idx]} | base prob {base:.3f}")

    results = []
    for gene_idx in shared_genes(data, drug_idx, disease_idx):
        edges = without_gene(data, drug_idx, disease_idx, gene_idx)
        new_prob = score_pair(model, data, drug_idx, disease_idx, edges)
        results.append((base - new_prob, names["gene"][gene_idx], new_prob))

    results.sort(reverse=True)
    for drop, gene, new_prob in results:
        print(f"  without {gene:10s}: prob {new_prob:.3f} (drop {drop:+.3f})")


def main(seed=42):
    torch.manual_seed(seed)
    names = load_names()
    _, _, test_data = load_data()

    model = BioGraphModel(test_data.metadata())
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location="cpu"))
    model.eval()

    drug_ids = {name: idx for idx, name in names["drug"].items()}
    disease_ids = {name: idx for idx, name in names["disease"].items()}

    for drug, disease in PAIRS:
        explain_pair(model, test_data, names, drug_ids[drug], disease_ids[disease])
        print()


if __name__ == "__main__":
    main()