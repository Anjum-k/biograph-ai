"""FastAPI service exposing a /predict endpoint."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
import torch

from src.models.biograph_model import BioGraphModel
from src.utils.config import EXPERIMENTS_DIR, MODEL_CONFIG

app = FastAPI(title="BioGraph AI API")

CKPT_PATH = EXPERIMENTS_DIR / "best_model.pt"
_model = None


def get_model(metadata):
    global _model
    if _model is None:
        _model = BioGraphModel(
            MODEL_CONFIG["hidden_channels"],
            MODEL_CONFIG["out_channels"],
            MODEL_CONFIG["num_heads"],
            MODEL_CONFIG["num_layers"],
            metadata,
        )
        if CKPT_PATH.exists():
            _model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        _model.eval()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(drug: str, disease: str):
    # Plug in your loaded graph_data + name-to-index mappings here, then:
    # score = ... run model.forward(...) on the relevant indices ...
    return {
        "drug": drug,
        "disease": disease,
        "note": "Wire this endpoint to your trained graph_data + id mappings.",
    }
