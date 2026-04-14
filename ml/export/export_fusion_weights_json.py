"""
Export the distilled LSTM + PerHorizon BayCal weights to JSON for Flutter.

Loads:
  - v2_student_distilled.pt  (pure-observation LSTM, 215K params)
  - v2_perhz_cal_dist.pt     (5 per-horizon fusion BayCal models, 25K params)
  - stats.pt                 (normalization statistics)

Outputs:
  mobile/assets/models/fusion_weights.json
  mobile/assets/models/fusion_stats.json

The JSON structure mirrors the PyTorch state_dict, with LSTM weights in
PyTorch gate order [input, forget, cell, output] and Linear layers as
{w: [[...]], b: [...]}.
"""

import json
import logging
from pathlib import Path

import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from lstm_forecaster import LSTMForecaster
from bayesian_cal import PerHorizonBayCal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Try research-spike checkpoints first, then main project
SPIKE_CKPT = Path(__file__).parent.parent.parent.parent / "research-spike" / "checkpoints"
PROJ_CKPT = Path(__file__).parent.parent / "checkpoints"
SPIKE_DATA = Path(__file__).parent.parent.parent.parent / "research-spike" / "data"
PROJ_DATA = Path(__file__).parent.parent / "data" / "processed"
ASSETS_DIR = Path(__file__).parent.parent.parent / "mobile" / "assets" / "models"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]


def t2l(t: torch.Tensor):
    """Tensor to nested list."""
    return t.detach().cpu().tolist()


def find_checkpoint(name: str) -> Path:
    for d in [SPIKE_CKPT, PROJ_CKPT]:
        p = d / name
        if p.exists():
            log.info(f"  Found {name} at {d}")
            return p
    raise FileNotFoundError(f"Cannot find {name} in {SPIKE_CKPT} or {PROJ_CKPT}")


def find_stats() -> Path:
    for d in [SPIKE_DATA, PROJ_DATA]:
        p = d / "stats.pt"
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find stats.pt")


def export_lstm(ckpt_path: Path) -> dict:
    """Export distilled pure-observation LSTM weights."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Reconstruct model to verify
    model = LSTMForecaster(
        n_vars=6, hidden_dim=128, n_layers=2,
        n_horizons=5, lstm_lookback=12, include_gfs=False,
    )
    model.load_state_dict(sd)
    log.info(f"  LSTM: {sum(p.numel() for p in model.parameters()):,} params")

    # Extract LSTM layers
    lstm_layers = []
    for i in range(2):  # 2 layers
        lstm_layers.append({
            "W_ih": t2l(sd[f"lstm.weight_ih_l{i}"]),
            "W_hh": t2l(sd[f"lstm.weight_hh_l{i}"]),
            "b_ih": t2l(sd[f"lstm.bias_ih_l{i}"]),
            "b_hh": t2l(sd[f"lstm.bias_hh_l{i}"]),
        })

    # Extract prediction heads (fc_mu and fc_logvar)
    fc_mu = [
        {"w": t2l(sd["fc_mu.0.weight"]), "b": t2l(sd["fc_mu.0.bias"])},
        {"w": t2l(sd["fc_mu.2.weight"]), "b": t2l(sd["fc_mu.2.bias"])},
    ]
    fc_logvar = [
        {"w": t2l(sd["fc_logvar.0.weight"]), "b": t2l(sd["fc_logvar.0.bias"])},
        {"w": t2l(sd["fc_logvar.2.weight"]), "b": t2l(sd["fc_logvar.2.bias"])},
    ]

    return {
        "lstm": {"layers": lstm_layers},
        "fc_mu": fc_mu,
        "fc_logvar": fc_logvar,
        "meta": {
            "hidden_size": 128,
            "input_size": 6,
            "n_layers": 2,
            "lstm_lookback": 12,
            "include_gfs": False,
        },
    }


def export_perhz_baycal(ckpt_path: Path) -> dict:
    """Export 5 per-horizon BayCal fusion models."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = PerHorizonBayCal(n_vars=6, hidden_dim=64, n_horizons=5)
    model.load_state_dict(sd)
    log.info(f"  PerHzCal: {sum(p.numel() for p in model.parameters()):,} params")

    horizons_data = []
    for i in range(5):
        horizons_data.append({
            "horizon": HORIZONS[i],
            "alpha_net": [
                {"w": t2l(sd[f"alpha_nets.{i}.0.weight"]),
                 "b": t2l(sd[f"alpha_nets.{i}.0.bias"])},
                {"w": t2l(sd[f"alpha_nets.{i}.2.weight"]),
                 "b": t2l(sd[f"alpha_nets.{i}.2.bias"])},
            ],
            "bias_net": [
                {"w": t2l(sd[f"bias_nets.{i}.0.weight"]),
                 "b": t2l(sd[f"bias_nets.{i}.0.bias"])},
                {"w": t2l(sd[f"bias_nets.{i}.2.weight"]),
                 "b": t2l(sd[f"bias_nets.{i}.2.bias"])},
            ],
            "logvar_net": [
                {"w": t2l(sd[f"logvar_nets.{i}.0.weight"]),
                 "b": t2l(sd[f"logvar_nets.{i}.0.bias"])},
                {"w": t2l(sd[f"logvar_nets.{i}.2.weight"]),
                 "b": t2l(sd[f"logvar_nets.{i}.2.bias"])},
            ],
        })

    return {"horizons": horizons_data}


def main():
    log.info("Exporting fusion model weights...")

    lstm_ckpt = find_checkpoint("v2_student_distilled.pt")
    baycal_ckpt = find_checkpoint("v2_perhz_cal_dist.pt")

    log.info("Exporting LSTM...")
    lstm_data = export_lstm(lstm_ckpt)

    log.info("Exporting PerHorizon BayCal...")
    baycal_data = export_perhz_baycal(baycal_ckpt)

    output = {
        "lstm": lstm_data,
        "baycal": baycal_data,
        "horizons": HORIZONS,
        "meta": {
            "architecture": "distilled_lstm_perhz_baycal",
            "lstm_params": 215276,
            "baycal_params": 24890,
            "total_params": 240166,
            "n_horizons": 5,
            "n_vars": 6,
        },
    }

    out_path = ASSETS_DIR / "fusion_weights.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    size_kb = out_path.stat().st_size / 1024
    log.info(f"Saved weights to {out_path} ({size_kb:.1f} KB)")

    # Export stats
    stats_path_src = find_stats()
    stats = torch.load(stats_path_src, weights_only=False)
    stats_out = ASSETS_DIR / "fusion_stats.json"
    with open(stats_out, "w") as f:
        json.dump(stats, f)
    log.info(f"Saved stats to {stats_out}")


if __name__ == "__main__":
    main()
