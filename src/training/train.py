"""Training loop for BioGraphModel."""
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.biograph_model import BioGraphModel
from src.training.evaluate import evaluate_auc

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def train_model(train_data, val_data, metadata, num_epochs=100, lr=0.001,
                 hidden_channels=64, out_channels=32, num_heads=4, num_layers=2,
                 patience=20, verbose=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BioGraphModel(hidden_channels, out_channels, num_heads, num_layers, metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_data, val_data = train_data.to(device), val_data.to(device)

    best_val_auc, epochs_no_improve = 0.0, 0
    ckpt_path = EXPERIMENTS_DIR / "best_model.pt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        edge_label_index = train_data["drug", "treats", "disease"].edge_label_index
        edge_label = train_data["drug", "treats", "disease"].edge_label

        out = model(train_data.x_dict, train_data.edge_index_dict, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label.float())
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == num_epochs:
            val_auc = evaluate_auc(model, val_data)
            if verbose:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 5
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best Val AUC: {best_val_auc:.4f})")
                    break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model
