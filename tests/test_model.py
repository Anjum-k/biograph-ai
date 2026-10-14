import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.biograph_model import BioGraphModel


def test_model_forward_pass():
    metadata = (["drug", "disease"], [("drug", "treats", "disease")])
    model = BioGraphModel(32, 16, 2, 1, metadata)

    x_dict = {"drug": torch.randn(5, 32), "disease": torch.randn(5, 32)}
    edge_index_dict = {("drug", "treats", "disease"): torch.tensor([[0, 1], [0, 1]])}
    edge_label_index = torch.tensor([[0, 1], [0, 1]])

    out = model(x_dict, edge_index_dict, edge_label_index)
    assert out.shape[0] == 2
