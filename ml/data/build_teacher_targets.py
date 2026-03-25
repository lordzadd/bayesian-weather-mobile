"""
Pre-computes teacher model predictions on the full training + validation sets.

Saves teacher_mean and teacher_std tensors alongside the existing data,
so the distillation training loop can load them without running the
teacher forward pass at training time.

Usage:
    cd ml/
    python data/build_teacher_targets.py
"""

import logging
from pathlib import Path

import torch
import pyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.bma_teacher import BMATeacher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


def main():
    ckpt_path = CKPT_DIR / "bma_teacher.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No teacher checkpoint. Run train_teacher.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BMATeacher(n_features=6, n_temporal=4)
    model.load_state_dict(ckpt["model_state"])
    pyro.get_param_store().set_state(ckpt["pyro_params"])
    model.eval()
    log.info(f"Loaded teacher from epoch {ckpt['epoch']}, val ELBO {ckpt['val_loss']:.2f}")

    for split in ["train", "val"]:
        data = torch.load(PROC_DIR / f"{split}.pt", weights_only=True)
        gfs = data["gfs"]
        spatial = data["spatial"]
        temporal = data.get("temporal", torch.zeros(gfs.shape[0], 4))

        log.info(f"Computing teacher predictions for {split} ({len(gfs)} samples)...")

        # Process in batches to avoid OOM
        batch_size = 4096
        all_means, all_stds = [], []

        with torch.no_grad():
            for i in range(0, len(gfs), batch_size):
                g = gfs[i:i + batch_size]
                s = spatial[i:i + batch_size]
                t = temporal[i:i + batch_size]
                mean, std = model.predict(g, s, t)
                all_means.append(mean)
                all_stds.append(std)

        teacher_mean = torch.cat(all_means, dim=0)
        teacher_std = torch.cat(all_stds, dim=0)

        # Save augmented dataset
        data["teacher_mean"] = teacher_mean
        data["teacher_std"] = teacher_std
        torch.save(data, PROC_DIR / f"{split}.pt")

        log.info(f"  Saved teacher targets to {split}.pt "
                 f"(mean shape {teacher_mean.shape}, std shape {teacher_std.shape})")


if __name__ == "__main__":
    main()
