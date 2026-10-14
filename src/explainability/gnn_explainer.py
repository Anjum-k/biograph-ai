"""Wraps PyG's GNNExplainer to explain a single drug-disease prediction."""
from torch_geometric.explain import Explainer, GNNExplainer


def build_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="edge",
            return_type="probs",
        ),
    )


def explain_prediction(explainer, x_dict, edge_index_dict, drug_idx, disease_idx):
    explanation = explainer(
        x=x_dict, edge_index=edge_index_dict, index=(drug_idx, disease_idx)
    )
    # explanation.edge_mask -> importance score per edge
    # explanation.node_mask -> importance score per node feature
    return explanation
