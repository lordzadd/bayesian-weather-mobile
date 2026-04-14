"""
Trains the BMA bias-correction model on paired (GFS, METAR) data.

Usage:
    python -m training.train [--epochs 150] [--lr 1e-3] [--batch-size 512]

Automatically uses MPS (Apple Silicon GPU) when available.
"""

import argparse
import logging
from pathlib import Path

import torch
import pyro
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from training.bma_model import BMAModel, build_svi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
CKPT_DIR  = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str) -> TensorDataset:
    path = PROC_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data/collect_training_data.py then data/build_dataset.py first."
        )
    d = torch.load(path, weights_only=True)
    # d keys: gfs [N,6], spatial [N,2], temporal [N,4], obs [N,6]
    temporal = d.get("temporal", torch.zeros(d["gfs"].shape[0], 4))
    return TensorDataset(d["gfs"], d["spatial"], temporal, d["obs"])


def train(args: argparse.Namespace):
    pyro.clear_param_store()
    device = best_device()
    log.info(f"Device: {device}")

    train_ds = load_split("train")
    val_ds   = load_split("val")
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = BMAModel(n_features=6, hidden_dim=args.hidden_dim).to(device)
    svi   = build_svi(model, lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for gfs, spatial, temporal, obs in tqdm(train_loader, desc=f"Ep {epoch:3d}", leave=False):
            gfs      = gfs.to(device)
            spatial  = spatial.to(device)
            temporal = temporal.to(device)
            obs      = obs.to(device)
            train_loss += svi.step(gfs, spatial, obs, temporal)

        # --- validate (ELBO on val set, no parameter update) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gfs, spatial, temporal, obs in val_loader:
                gfs      = gfs.to(device)
                spatial  = spatial.to(device)
                temporal = temporal.to(device)
                obs      = obs.to(device)
                val_loss += svi.evaluate_loss(gfs, spatial, obs, temporal)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d} | train ELBO: {avg_train:10.2f} | val ELBO: {avg_val:10.2f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "pyro_params": pyro.get_param_store().get_state(),
                "val_loss":    best_val_loss,
                "args":        vars(args),
            }, CKPT_DIR / "bma_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    log.info(f"Done. Best val ELBO: {best_val_loss:.2f}  →  checkpoints/bma_best.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--patience",   type=int,   default=15, help="Early stop after N epochs without val improvement")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--hidden-dim", type=int,   default=64)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
