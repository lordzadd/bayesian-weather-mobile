"""
Preprocesses raw ERA5 NetCDF files into tensor datasets for BMA training.

Outputs normalized tensors aligned to the GFS 0.25-degree grid.
"""

import numpy as np
import xarray as xr
import torch
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE_STATS_FILE = PROCESSED_DIR / "normalization_stats.pt"


def load_era5_files() -> xr.Dataset:
    files = sorted(RAW_DIR.glob("era5_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {RAW_DIR}. Run download_era5.py first.")
    logger.info(f"Loading {len(files)} ERA5 files...")
    return xr.open_mfdataset(files, combine="by_coords")


def compute_normalization_stats(ds: xr.Dataset) -> dict:
    stats = {}
    for var in ds.data_vars:
        data = ds[var].values.astype(np.float32)
        stats[var] = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
        }
        logger.info(f"  {var}: mean={stats[var]['mean']:.4f}, std={stats[var]['std']:.4f}")
    return stats


def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (data - mean) / (std + 1e-8)


def build_tensors(ds: xr.Dataset, stats: dict) -> dict[str, torch.Tensor]:
    tensors = {}
    for var in ds.data_vars:
        raw = ds[var].values.astype(np.float32)
        normed = normalize(raw, stats[var]["mean"], stats[var]["std"])
        tensors[var] = torch.from_numpy(normed)
    return tensors


def main():
    ds = load_era5_files()
    logger.info("Computing normalization statistics...")
    stats = compute_normalization_stats(ds)
    torch.save(stats, VARIABLE_STATS_FILE)
    logger.info(f"Saved normalization stats to {VARIABLE_STATS_FILE}")

    logger.info("Building normalized tensors...")
    tensors = build_tensors(ds, stats)

    output_path = PROCESSED_DIR / "era5_tensors.pt"
    torch.save(tensors, output_path)
    logger.info(f"Saved processed tensors to {output_path}")


if __name__ == "__main__":
    main()
