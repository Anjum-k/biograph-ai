"""Central configuration for paths and model hyperparameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_GRAPH = ROOT_DIR / "data" / "graph"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

MODEL_CONFIG = dict(
    hidden_channels=64,
    out_channels=32,
    num_heads=4,
    num_layers=2,
)

TRAIN_CONFIG = dict(
    num_epochs=100,
    lr=0.001,
    patience=20,
)
