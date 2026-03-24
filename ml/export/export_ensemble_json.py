"""
Exports all ensemble model weights to a single JSON file for Flutter.

The Dart engine loads all N model weight sets and averages their predictions.

Output:
  mobile/assets/models/bma_weights.json  — contains list of model weight sets
  mobile/assets/models/bma_stats.json    — normalization stats (unchanged)
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

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
ASSETS_DIR = Path(__file__).parent.parent.parent / "mobile" / "assets" / "models"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().tolist()


def extract_weights(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_dim = ckpt.get("args", {}).get("hidden_dim", 64)
    model = BMAModel(n_features=6, hidden_dim=hidden_dim, n_temporal=4)
    model.load_state_dict(ckpt["model_state"])
    sd = model.state_dict()

    return {
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
        "seed": ckpt.get("seed"),
        "val_elbo": ckpt.get("val_loss"),
    }


def main():
    # Find all ensemble checkpoints
    ensemble_files = sorted(CKPT_DIR.glob("bma_ensemble_*.pt"))

    if not ensemble_files:
        # Fall back to single model
        single = CKPT_DIR / "bma_best.pt"
        if not single.exists():
            raise FileNotFoundError("No checkpoints found.")
        ensemble_files = [single]

    log.info(f"Exporting {len(ensemble_files)} model(s)")

    models = []
    for f in ensemble_files:
        log.info(f"  Loading {f.name}")
        models.append(extract_weights(f))

    output = {
        "ensemble": models,
        "meta": {
            "n_models": len(models),
            "n_features": 6,
            "hidden_dim": 64,
            "n_temporal": 4,
        },
    }

    # For backwards compatibility, also include the first model's weights
    # at the top level so the existing Dart engine can load without changes
    output["bias_net"] = models[0]["bias_net"]
    output["noise_net"] = models[0]["noise_net"]

    out_path = ASSETS_DIR / "bma_weights.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    log.info(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Export stats
    stats_pt = PROC_DIR / "stats.pt"
    if stats_pt.exists():
        stats = torch.load(stats_pt, weights_only=False)
        stats_path = ASSETS_DIR / "bma_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        log.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
