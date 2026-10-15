import torch
from torch_geometric import data

from src.data.split import TARGET_EDGE
from src.models.biograph_model import BioGraphModel
from src.training.train import load_data, CHECKPOINT_PATH



def build_known(train_data, val_data, test_data):
    known = {}
    pair_sets = [train_data[TARGET_EDGE].edge_index]
    for data in (val_data, test_data):
        label_index = data[TARGET_EDGE].edge_label_index
        labels = data[TARGET_EDGE].edge_label
        pair_sets.append(label_index[:, labels == 1])
    for pairs in pair_sets:
        for drug, disease in zip(pairs[0].tolist(), pairs[1].tolist()):
            known.setdefault(drug, set()).add(disease)
    return known


def rank_of(scores, target, known_ids):
    scores = scores.clone()
    for other in known_ids:
        if other != target:
            scores[other] = float("-inf")
    target_score = scores[target]
    higher = (scores > target_score).sum().item()
    ties = (scores == target_score).sum().item() - 1
    return 1 + higher + ties / 2


def summarize(name, ranks):
    ranks = torch.tensor(ranks)
    print(
        f"{name:16s} MRR {(1 / ranks).mean():.3f} | "
        f"Hits@1 {(ranks <= 1).float().mean():.3f} | "
        f"Hits@3 {(ranks <= 3).float().mean():.3f} | "
        f"Hits@10 {(ranks <= 10).float().mean():.3f}"
    )

def shared_gene_matrix(data, num_drugs, num_diseases, num_genes):
    drug_gene = torch.zeros(num_drugs, num_genes)
    edge_index = data[("drug", "targets", "gene")].edge_index
    drug_gene[edge_index[0], edge_index[1]] = 1.0

    disease_gene = torch.zeros(num_diseases, num_genes)
    edge_index = data[("disease", "associated_with", "gene")].edge_index
    disease_gene[edge_index[0], edge_index[1]] = 1.0

    return drug_gene @ disease_gene.T

def main(seed=42):
    torch.manual_seed(seed)
    train_data, val_data, test_data = load_data()
    known = build_known(train_data, val_data, test_data)

    folder = CHECKPOINT_PATH.parent
    models = {
        "GNN random-feat": load_model(folder / "final_model.pt", test_data, False),
        "GNN embeddings": load_model(folder / "final_model_emb.pt", test_data, True),
    }

    num_diseases = test_data["disease"].num_nodes
    all_diseases = torch.arange(num_diseases)
    disease_deg = torch.bincount(
        train_data[TARGET_EDGE].edge_index[1], minlength=num_diseases
    ).float()

    shared = shared_gene_matrix(
        test_data,
        test_data["drug"].num_nodes,
        num_diseases,
        test_data["gene"].num_nodes,
    )

    label_index = test_data[TARGET_EDGE].edge_label_index
    labels = test_data[TARGET_EDGE].edge_label

    ranks = {name: [] for name in models}
    ranks["disease-degree"] = []
    ranks["shared-genes"] = []
    ranks["shared+pop"] = []

    with torch.no_grad():
        zs = {
            name: model.encode(test_data.x_dict, test_data.edge_index_dict)
            for name, model in models.items()
        }
        for i in range(label_index.size(1)):
            if labels[i].item() != 1:
                continue
            drug = label_index[0, i].item()
            disease = label_index[1, i].item()
            pairs = torch.stack([torch.full((num_diseases,), drug), all_diseases])
            for name, model in models.items():
                scores = model.decode(zs[name], pairs)
                ranks[name].append(rank_of(scores, disease, known[drug]))
            ranks["disease-degree"].append(rank_of(disease_deg, disease, known[drug]))
            ranks["shared-genes"].append(rank_of(shared[drug], disease, known[drug]))
            ranks["shared+pop"].append(
                rank_of(shared[drug] + 1e-3 * torch.log1p(disease_deg), disease, known[drug])
            )

    print(f"{len(ranks['disease-degree'])} test positives, {num_diseases} candidate diseases")
    for name, r in ranks.items():
        summarize(name, r)

def load_model(path, data, use_embeddings):
    if use_embeddings:
        model = BioGraphModel.from_data(data)
    else:
        model = BioGraphModel(data.metadata())
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    main()