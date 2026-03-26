"""
Exports linear regression weights to JSON for the Dart inference engine.

Reads:  checkpoints/linear_best.json  (produced by training/train_linear.py)
Writes: mobile/assets/models/linear_weights.json

JSON structure (matches LinearDartEngine expectations):
  {
    "coefficients":  [[8 floats] × 6],   // one row per output variable
    "intercepts":    [6 floats],
    "residual_stds": [6 floats],          // per-variable σ (validation residual std)
    "meta":          { ... }
  }

The Dart engine reuses bma_stats.json for input/output normalization.
"""

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"
ASSETS_DIR = Path(__file__).parent.parent.parent / "mobile" / "assets" / "models"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def export():
    src = CKPT_DIR / "linear_best.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run training/train_linear.py first."
        )

    with open(src) as f:
        ckpt = json.load(f)

    # Validate expected keys
    for key in ("coefficients", "intercepts", "residual_stds", "meta"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint missing key: {key}")

    dst = ASSETS_DIR / "linear_weights.json"
    shutil.copy(src, dst)
    log.info(f"Exported → {dst}  ({dst.stat().st_size / 1024:.1f} KB)")

    meta = ckpt["meta"]
    log.info("Validation metrics:")
    for i, name in enumerate(meta.get("var_names", [str(i) for i in range(6)])):
        mae = meta["val_mae"][i]
        r2  = meta["val_r2"][i]
        log.info(f"  {name:14s}  MAE={mae:.4f}  R²={r2:.4f}")


if __name__ == "__main__":
    export()
