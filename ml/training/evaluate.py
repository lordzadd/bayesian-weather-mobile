"""
Evaluates trained BMA model against held-out station observations.

Reports:
  - MAE: BMA posterior mean vs. raw GFS proxy
  - MAE: BMA posterior mean vs. ground truth (station obs)
  - Calibration: % of observations within 1σ and 2σ posterior intervals
"""

import logging
from pathlib import Path

import torch
import numpy as np
import pyro

from bma_model import BMAModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

N_EVAL_SAMPLES = 500  # MC draws for posterior predictive


def load_model(checkpoint_path: Path, n_features: int = 6) -> BMAModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BMAModel(n_features=n_features)
    model.load_state_dict(ckpt["model_state"])
    pyro.get_param_store().set_state(ckpt["pyro_params"])
    model.eval()
    return model


def evaluate(model: BMAModel, gfs: torch.Tensor, spatial: torch.Tensor, obs: torch.Tensor):
    with torch.no_grad():
        post_mean, post_std = model.predict(gfs, spatial, n_samples=N_EVAL_SAMPLES)

    mae_gfs = (gfs - obs).abs().mean().item()
    mae_bma = (post_mean - obs).abs().mean().item()

    within_1sigma = ((obs - post_mean).abs() < post_std).float().mean().item()
    within_2sigma = ((obs - post_mean).abs() < 2 * post_std).float().mean().item()

    logger.info(f"MAE (raw GFS):    {mae_gfs:.4f}")
    logger.info(f"MAE (BMA posterior): {mae_bma:.4f}")
    logger.info(f"Improvement:      {(1 - mae_bma / mae_gfs) * 100:.1f}%")
    logger.info(f"Calibration 1σ:   {within_1sigma * 100:.1f}% (expected ~68%)")
    logger.info(f"Calibration 2σ:   {within_2sigma * 100:.1f}% (expected ~95%)")

    return {
        "mae_gfs": mae_gfs,
        "mae_bma": mae_bma,
        "cal_1sigma": within_1sigma,
        "cal_2sigma": within_2sigma,
    }


if __name__ == "__main__":
    ckpt = CHECKPOINT_DIR / "bma_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError("No checkpoint found. Run train.py first.")

    model = load_model(ckpt)

    tensors = torch.load(PROCESSED_DIR / "era5_tensors.pt")
    variables = ["2m_temperature", "surface_pressure",
                 "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "total_precipitation", "relative_humidity"]
    stacked = torch.stack([tensors[v] for v in variables], dim=-1)
    flat = stacked.reshape(-1, len(variables))

    # Use last 10% as eval set
    n_eval = len(flat) // 10
    gfs = flat[-n_eval:]
    spatial = torch.zeros(n_eval, 2)
    obs = flat[-n_eval:]  # Replace with real held-out METAR obs in production

    evaluate(model, gfs, spatial, obs)
