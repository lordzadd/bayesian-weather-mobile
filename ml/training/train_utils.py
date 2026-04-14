"""
Training utilities shared by all models.

All models output (mu, logvar) and are trained with Gaussian NLL loss.
Separate functions for base models (LSTM/PatchTST) vs calibration layer
since they have different forward signatures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def gaussian_nll(mu, logvar, target):
    """Gaussian negative log-likelihood loss, averaged over all dims."""
    return 0.5 * (logvar + (target - mu) ** 2 / logvar.exp()).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_loader(data, batch_size, shuffle, keys):
    tensors = [data[k] for k in keys]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


def train_base_model(
    model,
    train_data,
    val_data,
    name: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
):
    """Train an LSTM or PatchTST model. Returns trained model on CPU."""
    print(f"\n{'='*60}")
    print(f"Training {name}  ({count_params(model):,} params)")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5,
    )

    keys = ["obs_hist", "gfs_targets", "spatial", "temporal", "obs_targets"]
    train_loader = _make_loader(train_data, batch_size, shuffle=True, keys=keys)
    val_loader = _make_loader(val_data, batch_size, shuffle=False, keys=keys)

    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            obs_h, gfs_t, sp, tmp, obs_t = [x.to(device) for x in batch]
            mu, logvar = model(obs_h, gfs_t, sp, tmp)
            loss = gaussian_nll(mu, logvar, obs_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs_h, gfs_t, sp, tmp, obs_t = [x.to(device) for x in batch]
                mu, logvar = model(obs_h, gfs_t, sp, tmp)
                val_loss += gaussian_nll(mu, logvar, obs_t).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            flag = " *" if improved else ""
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.1e}{flag}")

        if wait >= patience:
            print(f"  Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    model = model.cpu()
    torch.save(best_state, CKPT_DIR / f"{name}.pt")
    print(f"  Saved checkpoint: {name}.pt  (val={best_val:.4f})")
    return model


def generate_predictions(model, data, device="cpu", batch_size=512):
    """Run base model on full dataset, return (mu, logvar) tensors on CPU."""
    model = model.to(device).eval()
    keys = ["obs_hist", "gfs_targets", "spatial", "temporal"]
    loader = _make_loader(data, batch_size, shuffle=False, keys=keys)

    all_mu, all_logvar = [], []
    with torch.no_grad():
        for batch in loader:
            obs_h, gfs_t, sp, tmp = [x.to(device) for x in batch]
            mu, logvar = model(obs_h, gfs_t, sp, tmp)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    model = model.cpu()
    return torch.cat(all_mu), torch.cat(all_logvar)


def train_cal_model(
    cal_model,
    base_mu_train,
    base_logvar_train,
    train_data,
    base_mu_val,
    base_logvar_val,
    val_data,
    name: str,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
):
    """Train a Bayesian calibration layer on base model predictions."""
    print(f"\n{'='*60}")
    print(f"Training {name}  ({count_params(cal_model):,} params)")
    print(f"{'='*60}")

    cal_model = cal_model.to(device)
    optimizer = torch.optim.Adam(cal_model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(
            base_mu_train, base_logvar_train,
            train_data["gfs_targets"], train_data["spatial"],
            train_data["temporal"], train_data["obs_targets"],
        ),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            base_mu_val, base_logvar_val,
            val_data["gfs_targets"], val_data["spatial"],
            val_data["temporal"], val_data["obs_targets"],
        ),
        batch_size=batch_size, shuffle=False,
    )

    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        cal_model.train()
        train_loss = 0.0
        for batch in train_loader:
            b_mu, b_lv, gfs, sp, tmp, obs_t = [x.to(device) for x in batch]
            mu, logvar = cal_model(b_mu, b_lv, gfs, sp, tmp)
            loss = gaussian_nll(mu, logvar, obs_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cal_model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        cal_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_mu, b_lv, gfs, sp, tmp, obs_t = [x.to(device) for x in batch]
                mu, logvar = cal_model(b_mu, b_lv, gfs, sp, tmp)
                val_loss += gaussian_nll(mu, logvar, obs_t).item()
        val_loss /= len(val_loader)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in cal_model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or improved:
            flag = " *" if improved else ""
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}{flag}")

        if wait >= patience:
            print(f"  Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    cal_model.load_state_dict(best_state)
    cal_model = cal_model.cpu()
    torch.save(best_state, CKPT_DIR / f"{name}.pt")
    print(f"  Saved checkpoint: {name}.pt  (val={best_val:.4f})")
    return cal_model


def train_distilled(
    student,
    teacher_mu_train,
    teacher_logvar_train,
    teacher_mu_val,
    teacher_logvar_val,
    train_data,
    val_data,
    name: str,
    alpha: float = 0.5,
    beta: float = 0.3,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
):
    """
    Knowledge distillation: train student to match teacher soft targets
    while also fitting observations.

    Loss = alpha * MSE(student_mu, teacher_mu)
         + beta  * MSE(student_logvar, teacher_logvar)
         + gamma * NLL(obs | student)
    """
    gamma = 1.0 - alpha - beta
    print(f"\n{'='*60}")
    print(f"Distilling {name}  ({count_params(student):,} params)")
    print(f"  alpha={alpha} (teacher mu)  beta={beta} (teacher var)  gamma={gamma:.1f} (obs)")
    print(f"{'='*60}")

    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_data["obs_hist"], train_data["gfs_targets"],
            train_data["spatial"], train_data["temporal"],
            train_data["obs_targets"],
            teacher_mu_train, teacher_logvar_train,
        ),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_data["obs_hist"], val_data["gfs_targets"],
            val_data["spatial"], val_data["temporal"],
            val_data["obs_targets"],
            teacher_mu_val, teacher_logvar_val,
        ),
        batch_size=batch_size, shuffle=False,
    )

    mse = nn.MSELoss()
    best_val = float("inf")
    wait = 0
    best_state = None

    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        for batch in train_loader:
            obs_h, gfs_t, sp, tmp, obs_t, t_mu, t_lv = [x.to(device) for x in batch]
            s_mu, s_lv = student(obs_h, gfs_t, sp, tmp)
            loss = (
                alpha * mse(s_mu, t_mu)
                + beta * mse(s_lv, t_lv)
                + gamma * gaussian_nll(s_mu, s_lv, obs_t)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs_h, gfs_t, sp, tmp, obs_t, t_mu, t_lv = [x.to(device) for x in batch]
                s_mu, s_lv = student(obs_h, gfs_t, sp, tmp)
                val_loss += gaussian_nll(s_mu, s_lv, obs_t).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            flag = " *" if improved else ""
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.1e}{flag}")

        if wait >= patience:
            print(f"  Early stop at epoch {epoch} (best val={best_val:.4f})")
            break

    student.load_state_dict(best_state)
    student = student.cpu()
    torch.save(best_state, CKPT_DIR / f"{name}.pt")
    print(f"  Saved checkpoint: {name}.pt  (val={best_val:.4f})")
    return student
