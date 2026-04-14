"""
Distills a trained teacher model into smaller student models (ensemble).

Loss = α * MSE(student_mean, teacher_mean)
     + β * MSE(student_std, teacher_std)
     + (1 - α - β) * ELBO(student, real_obs)

The teacher's soft targets provide smoother gradients than raw observations,
leading to better-calibrated uncertainty and more stable training.

Usage:
    cd ml/
    python -m training.distill [--n-models 5] [--alpha 0.5] [--beta 0.3]
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import pyro
from pyro.infer import Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from training.bma_model import BMAModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name):
    d = torch.load(PROC_DIR / f"{name}.pt", weights_only=True)
    temporal = d.get("temporal", torch.zeros(d["gfs"].shape[0], 4))
    teacher_mean = d.get("teacher_mean")
    teacher_std = d.get("teacher_std")

    if teacher_mean is None or teacher_std is None:
        raise FileNotFoundError(
            f"No teacher targets in {name}.pt. Run build_teacher_targets.py first."
        )

    return TensorDataset(
        d["gfs"], d["spatial"], temporal, d["obs"],
        teacher_mean, teacher_std,
    )


def distill_single(seed, args):
    """Train one student model via distillation."""
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

    # We use a plain PyTorch optimizer for the combined loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    alpha = args.alpha
    beta = args.beta
    gamma = 1.0 - alpha - beta  # weight on ELBO against real obs

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for gfs, spatial, temporal, obs, t_mean, t_std in tqdm(
            train_loader, desc=f"Seed {seed} Ep {epoch:3d}", leave=False
        ):
            gfs = gfs.to(device)
            spatial = spatial.to(device)
            temporal = temporal.to(device)
            obs = obs.to(device)
            t_mean = t_mean.to(device)
            t_std = t_std.to(device)

            optimizer.zero_grad()

            # Student forward pass
            s_mean, s_std = model.predict(gfs, spatial, temporal)

            # Distillation losses
            loss_mean = F.mse_loss(s_mean, t_mean)
            loss_std = F.mse_loss(s_std, t_std)

            # ELBO-like loss: negative log-likelihood of real obs under student posterior
            nll = -torch.distributions.Normal(s_mean, s_std).log_prob(obs).sum(-1).mean()

            loss = alpha * loss_mean + beta * loss_std + gamma * nll
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gfs, spatial, temporal, obs, t_mean, t_std in val_loader:
                gfs = gfs.to(device)
                spatial = spatial.to(device)
                temporal = temporal.to(device)
                obs = obs.to(device)
                t_mean = t_mean.to(device)
                t_std = t_std.to(device)

                s_mean, s_std = model.predict(gfs, spatial, temporal)
                loss_mean = F.mse_loss(s_mean, t_mean)
                loss_std = F.mse_loss(s_std, t_std)
                nll = -torch.distributions.Normal(s_mean, s_std).log_prob(obs).sum(-1).mean()
                val_loss += (alpha * loss_mean + beta * loss_std + gamma * nll).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  Seed {seed} | Ep {epoch:3d} | train: {avg_train:.4f} | val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "seed": seed,
                "args": vars(args),
                "distilled": True,
            }, CKPT_DIR / f"bma_ensemble_{seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"  Seed {seed} | Early stop at epoch {epoch}")
                break

    return best_val_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.5, help="Weight on teacher mean matching")
    p.add_argument("--beta", type=float, default=0.3, help="Weight on teacher std matching")
    args = p.parse_args()

    seeds = list(range(42, 42 + args.n_models))
    results = []

    for i, seed in enumerate(seeds):
        log.info(f"\n{'='*60}")
        log.info(f"Distilling model {i+1}/{args.n_models} (seed={seed})")
        log.info(f"  α={args.alpha} (teacher mean), β={args.beta} (teacher std), "
                 f"γ={1-args.alpha-args.beta:.1f} (real obs)")
        log.info(f"{'='*60}")
        val_loss = distill_single(seed, args)
        results.append((seed, val_loss))
        log.info(f"  Model {i+1} best val loss: {val_loss:.4f}")

    # Save best as bma_best.pt
    import shutil
    best_seed, best_loss = min(results, key=lambda x: x[1])
    shutil.copy(CKPT_DIR / f"bma_ensemble_{best_seed}.pt", CKPT_DIR / "bma_best.pt")

    log.info(f"\n{'='*60}")
    log.info(f"Distillation complete!")
    for seed, loss in results:
        log.info(f"  Seed {seed}: val loss = {loss:.4f}")
    log.info(f"  Best: seed {best_seed} ({best_loss:.4f}) → bma_best.pt")


if __name__ == "__main__":
    main()
