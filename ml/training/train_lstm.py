"""
Trains the LSTM sequence forecaster on paired (GFS, ERA5) data.

Sequences are built as sliding windows of length seq_len over the flat
training tensors. A small fraction of windows may cross station boundaries;
this adds noise but does not break training given the dataset size.

Loss: NLL under Normal(mean, std) — equivalent to MSE + uncertainty calibration.

Usage:
    python -m training.train_lstm [--epochs 100] [--hidden 64] [--seq-len 6]
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.lstm_model import LSTMForecaster

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


class SequenceDataset(Dataset):
    """Sliding-window sequences over the flat hourly tensor."""

    def __init__(self, name: str, seq_len: int):
        path = PROC_DIR / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run data/build_dataset.py first.")
        d = torch.load(path, weights_only=True)
        gfs = d["gfs"]         # [N, 6]
        spatial = d["spatial"]  # [N, 2]
        self.obs = d["obs"]     # [N, 6]
        self.X = torch.cat([gfs, spatial], dim=1)  # [N, 8]
        self.seq_len = seq_len
        self.n = len(self.X) - seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # sequence: [seq_len, 8], target: obs at idx + seq_len
        x_seq = self.X[idx : idx + self.seq_len]       # [seq_len, 8]
        y = self.obs[idx + self.seq_len]                # [6]
        return x_seq, y


def train(args: argparse.Namespace):
    device = best_device()
    log.info(f"Device: {device}")

    train_ds = SequenceDataset("train", args.seq_len)
    val_ds   = SequenceDataset("val",   args.seq_len)
    log.info(f"Train sequences: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = LSTMForecaster(
        input_size=8,
        hidden_size=args.hidden,
        num_layers=2,
        output_size=6,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_seq, y in train_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            mean, std = model(x_seq)
            loss = -torch.distributions.Normal(mean, std).log_prob(y).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, y in val_loader:
                x_seq, y = x_seq.to(device), y.to(device)
                mean, std = model(x_seq)
                val_loss += -torch.distributions.Normal(mean, std).log_prob(y).mean().item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d} | train NLL: {avg_train:.4f} | val NLL: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, CKPT_DIR / "lstm_best.pt")

    log.info(f"Done. Best val NLL: {best_val_loss:.4f}  →  checkpoints/lstm_best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--seq-len",    type=int,   default=6)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
