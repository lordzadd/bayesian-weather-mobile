"""
Training loop for the BMA model using Stochastic Variational Inference.

Usage:
    python training/train.py [--epochs 200] [--lr 1e-3] [--batch-size 256]
"""

import argparse
import logging
from pathlib import Path

import torch
import pyro
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bma_model import BMAModel, build_svi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_dataset() -> TensorDataset:
    tensors_path = PROCESSED_DIR / "era5_tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError("Processed tensors not found. Run data/preprocess.py first.")

    data = torch.load(tensors_path)
    variables = ["2m_temperature", "surface_pressure",
                 "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "total_precipitation", "relative_humidity"]

    # Stack variables along feature dim; flatten spatial/time dims into batch
    stacked = torch.stack([data[v] for v in variables], dim=-1)
    original_shape = stacked.shape
    flat = stacked.reshape(-1, len(variables))

    # Dummy spatial embeddings (lat/lon normalized to [-1, 1])
    # Replace with actual coordinate tensors from ERA5 grid in production
    spatial = torch.zeros(flat.shape[0], 2)

    logger.info(f"Dataset: {flat.shape[0]} samples, {len(variables)} features (from {original_shape})")
    return TensorDataset(flat, spatial, flat)  # (gfs_proxy, spatial, target)


def train(args: argparse.Namespace):
    pyro.clear_param_store()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = BMAModel(n_features=6, hidden_dim=64).to(device)
    svi = build_svi(model, lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for gfs, spatial, obs in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            gfs, spatial, obs = gfs.to(device), spatial.to(device), obs.to(device)
            loss = svi.step(gfs, spatial, obs)
            epoch_loss += loss

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch:4d} | ELBO loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = CHECKPOINT_DIR / "bma_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "pyro_params": pyro.get_param_store().get_state(),
                "loss": best_loss,
            }, ckpt_path)
            logger.info(f"  -> Saved best checkpoint to {ckpt_path}")

    logger.info(f"Training complete. Best ELBO loss: {best_loss:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BMA weather model")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
