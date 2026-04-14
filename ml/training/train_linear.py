"""
Ridge regression bias-correction baseline.

Input:  X = concat(gfs_norm[6], spatial[2]) — 8 features (same as BMA input)
Output: y = obs_norm[6] — normalized ERA5 targets

Trains one Ridge regressor per output variable, reports per-variable MAE and R²
on the validation split, then saves coefficients + residual stds to a checkpoint.

Usage:
    python -m training.train_linear [--alpha 1.0]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

VAR_NAMES = ["temp_c", "pressure_hpa", "u_ms", "v_ms", "precip_mm", "rh_pct"]


def load_split(name: str):
    path = PROC_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data/build_dataset.py first."
        )
    d = torch.load(path, weights_only=True)
    gfs = d["gfs"].numpy()        # [N, 6] normalized
    spatial = d["spatial"].numpy()  # [N, 2] scaled lat/lon
    obs = d["obs"].numpy()         # [N, 6] normalized
    X = np.concatenate([gfs, spatial], axis=1)  # [N, 8]
    return X, obs


def train(alpha: float):
    log.info("Loading data …")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    log.info(f"Train: {X_train.shape}  Val: {X_val.shape}")

    coefs = []
    intercepts = []
    residual_stds = []
    val_maes = []
    val_r2s = []

    for i, name in enumerate(VAR_NAMES):
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train[:, i])

        y_pred = reg.predict(X_val)
        mae = mean_absolute_error(y_val[:, i], y_pred)
        r2 = r2_score(y_val[:, i], y_pred)
        res_std = float(np.std(y_val[:, i] - y_pred))

        log.info(f"  {name:14s}  MAE={mae:.4f}  R²={r2:.4f}  residual_std={res_std:.4f}")

        coefs.append(reg.coef_.tolist())
        intercepts.append(float(reg.intercept_))
        residual_stds.append(res_std)
        val_maes.append(float(mae))
        val_r2s.append(float(r2))

    checkpoint = {
        "coefficients": coefs,       # [6][8]
        "intercepts": intercepts,    # [6]
        "residual_stds": residual_stds,  # [6] — used as σ by Dart engine
        "meta": {
            "alpha": alpha,
            "n_inputs": 8,
            "n_outputs": 6,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "val_mae": val_maes,
            "val_r2": val_r2s,
            "var_names": VAR_NAMES,
        },
    }

    out = CKPT_DIR / "linear_best.json"
    with open(out, "w") as f:
        json.dump(checkpoint, f)
    log.info(f"Saved → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularisation strength (default: 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.alpha)
