"""
Exports LSTM model weights to JSON for the Dart inference engine.

Reads:  checkpoints/lstm_best.pt  (produced by training/train_lstm.py)
Writes: mobile/assets/models/lstm_weights.json

JSON structure (matches LstmDartEngine expectations):
  {
    "lstm": {
      "layers": [
        {
          "W_ih": [[hidden×4, input_size]],   // input-hidden weights
          "W_hh": [[hidden×4, hidden_size]],  // hidden-hidden weights
          "b_ih": [hidden×4],
          "b_hh": [hidden×4]
        },
        ...                                   // one entry per LSTM layer
      ]
    },
    "mean_head": { "w": [[6, hidden]], "b": [6] },
    "std_head":  { "w": [[6, hidden]], "b": [6] },
    "meta": {
      "hidden_size": 64,
      "num_layers": 2,
      "input_size": 8,
      "n_outputs": 6,
      "seq_len": 6,
      "val_nll": float
    }
  }

The Dart engine reuses bma_stats.json for input/output normalization.
"""

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.lstm_model import LSTMForecaster

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"
ASSETS_DIR = Path(__file__).parent.parent.parent / "mobile" / "assets" / "models"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def t2l(t: torch.Tensor):
    return t.detach().cpu().tolist()


def export():
    ckpt_path = CKPT_DIR / "lstm_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run training/train_lstm.py first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

    hidden  = saved_args.get("hidden",  64)
    seq_len = saved_args.get("seq_len", 6)

    model = LSTMForecaster(input_size=8, hidden_size=hidden, num_layers=2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sd = model.state_dict()

    lstm_layers = []
    for layer_idx in range(model.num_layers):
        lstm_layers.append({
            "W_ih": t2l(sd[f"lstm.weight_ih_l{layer_idx}"]),
            "W_hh": t2l(sd[f"lstm.weight_hh_l{layer_idx}"]),
            "b_ih": t2l(sd[f"lstm.bias_ih_l{layer_idx}"]),
            "b_hh": t2l(sd[f"lstm.bias_hh_l{layer_idx}"]),
        })

    weights = {
        "lstm": {"layers": lstm_layers},
        "mean_head": {
            "w": t2l(sd["mean_head.weight"]),
            "b": t2l(sd["mean_head.bias"]),
        },
        "std_head": {
            "w": t2l(sd["log_std_head.weight"]),
            "b": t2l(sd["log_std_head.bias"]),
        },
        "meta": {
            "hidden_size": hidden,
            "num_layers":  model.num_layers,
            "input_size":  8,
            "n_outputs":   6,
            "seq_len":     seq_len,
            "val_nll":     float(ckpt["val_loss"]),
            "epoch":       ckpt["epoch"],
        },
    }

    out = ASSETS_DIR / "lstm_weights.json"
    with open(out, "w") as f:
        json.dump(weights, f)

    log.info(f"Exported → {out}  ({out.stat().st_size / 1024:.1f} KB)")
    log.info(f"  hidden={hidden}  layers={model.num_layers}  seq_len={seq_len}")
    log.info(f"  val NLL: {ckpt['val_loss']:.4f}  (epoch {ckpt['epoch']})")


if __name__ == "__main__":
    export()
