"""
Trains an ensemble of N BMA models with different random seeds.

Each model sees the same data but learns slightly different bias corrections
due to different weight initializations and stochastic optimization paths.
At inference time, predictions are averaged for better calibration.

Usage:
    cd ml/
    python -m training.train_ensemble [--n-models 5] [--epochs 150]
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

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str) -> TensorDataset:
    d = torch.load(PROC_DIR / f"{name}.pt", weights_only=True)
    temporal = d.get("temporal", torch.zeros(d["gfs"].shape[0], 4))
    return TensorDataset(d["gfs"], d["spatial"], temporal, d["obs"])


def train_single(seed: int, args: argparse.Namespace) -> float:
    """Train one model with a specific seed. Returns best val loss."""
    torch.manual_seed(seed)
    pyro.clear_param_store()
    device = best_device()

    train_ds = load_split("train")
    val_ds = load_split("val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = BMAModel(n_features=6, hidden_dim=args.hidden_dim, n_temporal=4).to(device)
    svi = build_svi(model, lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for gfs, spatial, temporal, obs in tqdm(train_loader,
                                                 desc=f"Seed {seed} Ep {epoch:3d}",
                                                 leave=False):
            gfs = gfs.to(device)
            spatial = spatial.to(device)
            temporal = temporal.to(device)
            obs = obs.to(device)
            train_loss += svi.step(gfs, spatial, obs, temporal)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gfs, spatial, temporal, obs in val_loader:
                gfs = gfs.to(device)
                spatial = spatial.to(device)
                temporal = temporal.to(device)
                obs = obs.to(device)
                val_loss += svi.evaluate_loss(gfs, spatial, obs, temporal)

        avg_val = val_loss / len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            avg_train = train_loss / len(train_loader)
            log.info(f"  Seed {seed} | Ep {epoch:3d} | train: {avg_train:10.2f} | val: {avg_val:10.2f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "pyro_params": pyro.get_param_store().get_state(),
                "val_loss": best_val_loss,
                "seed": seed,
                "args": vars(args),
            }, CKPT_DIR / f"bma_ensemble_{seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"  Seed {seed} | Early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    return best_val_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15, help="Early stop after N epochs without val improvement")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64)
    args = p.parse_args()

    seeds = list(range(42, 42 + args.n_models))
    results = []

    for i, seed in enumerate(seeds):
        log.info(f"\n{'='*60}")
        log.info(f"Training model {i+1}/{args.n_models} (seed={seed})")
        log.info(f"{'='*60}")
        val_loss = train_single(seed, args)
        results.append((seed, val_loss))
        log.info(f"  Model {i+1} best val ELBO: {val_loss:.2f}")

    # Also save the best single model as bma_best.pt for backwards compatibility
    best_seed, best_loss = min(results, key=lambda x: x[1])
    import shutil
    shutil.copy(CKPT_DIR / f"bma_ensemble_{best_seed}.pt", CKPT_DIR / "bma_best.pt")

    log.info(f"\n{'='*60}")
    log.info(f"Ensemble training complete!")
    log.info(f"{'='*60}")
    for seed, loss in results:
        log.info(f"  Seed {seed}: val ELBO = {loss:.2f}")
    log.info(f"  Best: seed {best_seed} ({best_loss:.2f}) → bma_best.pt")


if __name__ == "__main__":
    main()
