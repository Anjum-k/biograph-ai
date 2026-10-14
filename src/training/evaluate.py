"""Evaluation metrics: AUROC, AUPRC, and a simple Hits@K / MRR ranking eval."""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def evaluate_auc(model, data) -> float:
    model.eval()
    edge_label_index = data["drug", "treats", "disease"].edge_label_index
    edge_label = data["drug", "treats", "disease"].edge_label
    out = model(data.x_dict, data.edge_index_dict, edge_label_index)
    pred = torch.sigmoid(out).cpu().numpy()
    return roc_auc_score(edge_label.cpu().numpy(), pred)


@torch.no_grad()
def full_evaluation(model, data) -> dict:
    model.eval()
    edge_label_index = data["drug", "treats", "disease"].edge_label_index
    edge_label = data["drug", "treats", "disease"].edge_label
    out = model(data.x_dict, data.edge_index_dict, edge_label_index)
    pred = torch.sigmoid(out).cpu().numpy()
    labels = edge_label.cpu().numpy()

    return {
        "AUROC": roc_auc_score(labels, pred),
        "AUPRC": average_precision_score(labels, pred),
    }


@torch.no_grad()
def hits_at_k_and_mrr(model, data, num_diseases: int, k_values=(1, 3, 10)) -> dict:
    """
    For every positive drug-disease pair in `data`, rank the true disease
    against ALL diseases (full candidate ranking) -- the standard KG
    completion benchmark protocol.
    """
    model.eval()
    out_dict = model.encode(data.x_dict, data.edge_index_dict)
    drug_emb_all = out_dict["drug"]
    disease_emb_all = out_dict["disease"]

    edge_label_index = data["drug", "treats", "disease"].edge_label_index
    edge_label = data["drug", "treats", "disease"].edge_label
    pos_mask = edge_label == 1
    pos_pairs = edge_label_index[:, pos_mask]

    ranks = []
    for drug_idx, true_disease_idx in zip(pos_pairs[0].tolist(), pos_pairs[1].tolist()):
        drug_emb = drug_emb_all[drug_idx]
        scores = model.decoder(
            drug_emb.unsqueeze(0).expand(num_diseases, -1), disease_emb_all
        )
        scores = torch.sigmoid(scores).cpu().numpy()
        rank = (scores >= scores[true_disease_idx]).sum()
        ranks.append(rank)

    ranks = np.array(ranks)
    results = {"MRR": float(np.mean(1.0 / ranks))}
    for k in k_values:
        results[f"Hits@{k}"] = float(np.mean(ranks <= k))
    return results
