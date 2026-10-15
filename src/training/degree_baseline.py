import numpy as np
import torch

from src.data.split import TARGET_EDGE
from src.training.train import load_data, roc_auc, avg_precision


def degree(edge_index, num_nodes, row):
    return torch.bincount(edge_index[row], minlength=num_nodes).numpy()


def main(seeds=(42, 43, 44, 45, 46)):
    aurocs, auprcs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        train_data, val_data, test_data = load_data()

        edge_index = train_data[TARGET_EDGE].edge_index
        drug_deg = degree(edge_index, train_data["drug"].num_nodes, 0)
        disease_deg = degree(edge_index, train_data["disease"].num_nodes, 1)

        label_index = test_data[TARGET_EDGE].edge_label_index
        labels = test_data[TARGET_EDGE].edge_label.numpy()
        drugs = label_index[0].numpy()
        diseases = label_index[1].numpy()

        scores = np.log1p(drug_deg[drugs]) + np.log1p(disease_deg[diseases])
        auroc = roc_auc(labels, scores)
        auprc = avg_precision(labels, scores)
        aurocs.append(auroc)
        auprcs.append(auprc)
        print(f"seed {seed}: degree baseline AUROC {auroc:.4f} | AUPRC {auprc:.4f}")

    print(f"\nMEAN AUROC {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"MEAN AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")


if __name__ == "__main__":
    main()