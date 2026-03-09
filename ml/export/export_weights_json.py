"""
Exports BMA model weights to JSON so the Flutter Dart engine can run
the neural network forward pass without ExecuTorch.

Architecture exported:
  bias_net:  Linear(8→64) → ReLU → Linear(64→64) → ReLU → Linear(64→6)
  noise_net: Linear(8→32) → ReLU → Linear(32→6)  → Softplus

Output:
  mobile/assets/models/bma_weights.json
  mobile/assets/models/bma_stats.json  (normalization stats)
"""

import json
import logging
from pathlib import Path

import torch
import pyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.bma_model import BMAModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"
PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
ASSETS_DIR = Path(__file__).parent.parent.parent / "mobile" / "assets" / "models"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().tolist()


def export_weights(ckpt_path: Path, out_path: Path):
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = BMAModel(n_features=6, hidden_dim=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sd = model.state_dict()

    # bias_net layers: indices 0, 2, 4 in the Sequential
    weights = {
        "bias_net": [
            {"w": tensor_to_list(sd["bias_net.0.weight"]),
             "b": tensor_to_list(sd["bias_net.0.bias"])},
            {"w": tensor_to_list(sd["bias_net.2.weight"]),
             "b": tensor_to_list(sd["bias_net.2.bias"])},
            {"w": tensor_to_list(sd["bias_net.4.weight"]),
             "b": tensor_to_list(sd["bias_net.4.bias"])},
        ],
        "noise_net": [
            {"w": tensor_to_list(sd["noise_net.0.weight"]),
             "b": tensor_to_list(sd["noise_net.0.bias"])},
            {"w": tensor_to_list(sd["noise_net.2.weight"]),
             "b": tensor_to_list(sd["noise_net.2.bias"])},
        ],
        "meta": {
            "n_features": 6,
            "hidden_dim": 64,
            "epoch": ckpt["epoch"],
            "val_elbo": ckpt["val_loss"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(weights, f)
    log.info(f"Saved weights to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


def export_stats(stats_path: Path, out_path: Path):
    stats = torch.load(stats_path, weights_only=False)
    # stats is a dict: col -> {mean, std}
    with open(out_path, "w") as f:
        json.dump(stats, f)
    log.info(f"Saved stats to {out_path}")


if __name__ == "__main__":
    ckpt = CKPT_DIR / "bma_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError("Run ml/training/train.py first.")

    export_weights(ckpt, ASSETS_DIR / "bma_weights.json")

    stats_pt = PROC_DIR / "stats.pt"
    if stats_pt.exists():
        export_stats(stats_pt, ASSETS_DIR / "bma_stats.json")
