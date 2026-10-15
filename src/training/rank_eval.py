import torch

from src.data.split import TARGET_EDGE
from src.models.biograph_model import BioGraphModel
from src.training.train import load_data, CHECKPOINT_PATH

FINAL_MODEL_PATH = CHECKPOINT_PATH.parent / "final_model.pt"


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


def main(seed=42):
    torch.manual_seed(seed)
    train_data, val_data, test_data = load_data()
    known = build_known(train_data, val_data, test_data)

    model = BioGraphModel(test_data.metadata())
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location="cpu"))
    model.eval()

    num_diseases = test_data["disease"].num_nodes
    all_diseases = torch.arange(num_diseases)
    disease_deg = torch.bincount(
        train_data[TARGET_EDGE].edge_index[1], minlength=num_diseases
    ).float()

    label_index = test_data[TARGET_EDGE].edge_label_index
    labels = test_data[TARGET_EDGE].edge_label

    model_ranks, baseline_ranks = [], []
    with torch.no_grad():
        z = model.encode(test_data.x_dict, test_data.edge_index_dict)
        for i in range(label_index.size(1)):
            if labels[i].item() != 1:
                continue
            drug = label_index[0, i].item()
            disease = label_index[1, i].item()
            pairs = torch.stack([torch.full((num_diseases,), drug), all_diseases])
            scores = model.decode(z, pairs)
            model_ranks.append(rank_of(scores, disease, known[drug]))
            baseline_ranks.append(rank_of(disease_deg, disease, known[drug]))

    print(f"{len(model_ranks)} test positives, {num_diseases} candidate diseases")
    summarize("GNN (seed 42)", model_ranks)
    summarize("disease-degree", baseline_ranks)


if __name__ == "__main__":
    main()