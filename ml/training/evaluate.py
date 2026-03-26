"""
Evaluates trained BMA model against held-out validation data.

Reports per-variable and aggregate:
  - MAE: raw GFS vs. ground truth (baseline — what you get without BMA)
  - MAE: BMA posterior mean vs. ground truth (our model)
  - Improvement: % reduction in MAE over raw GFS
  - Calibration: % of observations within 1σ and 2σ posterior intervals

Usage:
    cd ml/
    python -m training.evaluate
"""

import logging
from pathlib import Path

import torch
import pyro

from training.bma_model import BMAModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

FEATURE_NAMES = [
    "temperature (°C)",
    "pressure (hPa)",
    "u-wind (m/s)",
    "v-wind (m/s)",
    "precipitation (mm)",
    "humidity (%)",
]


def load_model(checkpoint_path: Path, n_features: int = 6) -> BMAModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hidden_dim = ckpt.get("args", {}).get("hidden_dim", 64)
    model = BMAModel(n_features=n_features, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    if "pyro_params" in ckpt:
        pyro.get_param_store().set_state(ckpt["pyro_params"])
    model.eval()
    return model


def load_stats() -> dict:
    """Load normalization stats so we can report errors in physical units."""
    stats_path = PROCESSED_DIR / "stats.pt"
    if stats_path.exists():
        return torch.load(stats_path, weights_only=False)
    return None


def evaluate(model: BMAModel, gfs: torch.Tensor, spatial: torch.Tensor, obs: torch.Tensor, stats: dict = None, temporal: torch.Tensor = None):
    with torch.no_grad():
        post_mean, post_std = model.predict(gfs, spatial, temporal)

    # --- Aggregate metrics (normalized space) ---
    mae_gfs_norm = (gfs[:, :6] - obs).abs().mean().item()
    mae_bma_norm = (post_mean - obs).abs().mean().item()

    logger.info("=" * 60)
    logger.info("AGGREGATE METRICS (normalized space)")
    logger.info("=" * 60)
    logger.info(f"  MAE (raw GFS):       {mae_gfs_norm:.4f}")
    logger.info(f"  MAE (BMA posterior):  {mae_bma_norm:.4f}")
    if mae_gfs_norm > 0:
        logger.info(f"  Improvement:         {(1 - mae_bma_norm / mae_gfs_norm) * 100:.1f}%")

    # --- Calibration ---
    within_1sigma = ((obs - post_mean).abs() < post_std).float().mean().item()
    within_2sigma = ((obs - post_mean).abs() < 2 * post_std).float().mean().item()
    logger.info(f"  Calibration 1σ:      {within_1sigma * 100:.1f}% (expected ~68%)")
    logger.info(f"  Calibration 2σ:      {within_2sigma * 100:.1f}% (expected ~95%)")

    # --- Per-variable metrics ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("PER-VARIABLE METRICS")
    logger.info("=" * 60)

    gfs_cols = ["gfs_temp_c", "gfs_pres_hpa", "gfs_u_ms", "gfs_v_ms", "gfs_precip_mm", "gfs_rh_pct"]
    obs_cols = ["obs_temp_c", "obs_pres_hpa", "obs_u_ms", "obs_v_ms", "obs_precip_mm", "obs_rh_pct"]

    for i, name in enumerate(FEATURE_NAMES):
        gfs_err = (gfs[:, i] - obs[:, i]).abs().mean().item()
        bma_err = (post_mean[:, i] - obs[:, i]).abs().mean().item()
        cal_1s = ((obs[:, i] - post_mean[:, i]).abs() < post_std[:, i]).float().mean().item()
        cal_2s = ((obs[:, i] - post_mean[:, i]).abs() < 2 * post_std[:, i]).float().mean().item()
        improvement = (1 - bma_err / gfs_err) * 100 if gfs_err > 0 else 0

        # Denormalize errors to physical units if stats available
        phys_label = ""
        if stats is not None and i < len(obs_cols):
            col = obs_cols[i]
            if col in stats:
                scale = stats[col]["std"]
                phys_gfs = gfs_err * scale
                phys_bma = bma_err * scale
                phys_label = f"  [physical: GFS {phys_gfs:.3f}, BMA {phys_bma:.3f}]"

        logger.info(f"  {name:22s}  GFS={gfs_err:.4f}  BMA={bma_err:.4f}  "
                     f"Δ={improvement:+.1f}%  1σ={cal_1s*100:.0f}%  2σ={cal_2s*100:.0f}%{phys_label}")

    return {
        "mae_gfs": mae_gfs_norm,
        "mae_bma": mae_bma_norm,
        "cal_1sigma": within_1sigma,
        "cal_2sigma": within_2sigma,
    }


if __name__ == "__main__":
    ckpt = CHECKPOINT_DIR / "bma_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError("No checkpoint found. Run train.py first.")

    model = load_model(ckpt)

    val_path = PROCESSED_DIR / "val.pt"
    if not val_path.exists():
        raise FileNotFoundError("No val.pt found. Run build_dataset.py first.")

    val = torch.load(val_path, weights_only=True)
    gfs = val["gfs"]
    spatial = val["spatial"]
    temporal = val.get("temporal")
    obs = val["obs"]

    logger.info(f"Evaluating on {len(gfs)} validation samples")
    logger.info(f"Checkpoint: {ckpt}")
    logger.info("")

    stats = load_stats()
    evaluate(model, gfs, spatial, obs, stats, temporal)
