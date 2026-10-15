import torch

from src.data.split import TARGET_EDGE
from src.explainability.pair_explainer import load_names, shared_genes
from src.training.rank_eval import build_known, load_model, shared_gene_matrix
from src.training.train import load_data, CHECKPOINT_PATH


class BioGraphService:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        self.names = load_names()
        train_data, val_data, test_data = load_data()
        self.data = test_data
        self.known = build_known(train_data, val_data, test_data)

        model_path = CHECKPOINT_PATH.parent / "final_model.pt"
        self.model = load_model(model_path, test_data, False)

        self.num_diseases = test_data["disease"].num_nodes
        self.disease_deg = torch.bincount(
            train_data[TARGET_EDGE].edge_index[1], minlength=self.num_diseases
        ).float()
        self.shared = shared_gene_matrix(
            test_data,
            test_data["drug"].num_nodes,
            self.num_diseases,
            test_data["gene"].num_nodes,
        )
        with torch.no_grad():
            self.z = self.model.encode(test_data.x_dict, test_data.edge_index_dict)

        self.drug_index = {name: idx for idx, name in self.names["drug"].items()}

    def drug_names(self):
        return sorted(self.drug_index)

    def rank_diseases(self, drug_name, top_n=10, include_known=False):
        drug = self.drug_index[drug_name]
        known = self.known.get(drug, set())
        score = self.shared[drug] + 1e-3 * torch.log1p(self.disease_deg)

        pairs = torch.stack(
            [torch.full((self.num_diseases,), drug), torch.arange(self.num_diseases)]
        )
        with torch.no_grad():
            probs = torch.sigmoid(self.model.decode(self.z, pairs))

        results = []
        for disease in torch.argsort(score, descending=True).tolist():
            is_known = disease in known
            if is_known and not include_known:
                continue
            genes = shared_genes(self.data, drug, disease)
            results.append(
                {
                    "disease": self.names["disease"][disease],
                    "shared_gene_count": len(genes),
                    "gnn_prob": probs[disease].item(),
                    "shared_genes": [self.names["gene"][g] for g in genes],
                    "known": is_known,
                }
            )
            if len(results) == top_n:
                break
        return results