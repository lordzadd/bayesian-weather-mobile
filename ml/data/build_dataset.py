"""
Joins GFS proxy + METAR observations on (station, time), cleans, and
saves aligned tensors for training.

Output:
  ml/data/processed/train.pt   — 80% split
  ml/data/processed/val.pt     — 20% split
  ml/data/processed/stats.pt   — normalization mean/std per feature
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

GFS_FEATURES = ["gfs_temp_c","gfs_pres_hpa","gfs_u_ms","gfs_v_ms","gfs_precip_mm","gfs_rh_pct"]
OBS_FEATURES = ["obs_temp_c","obs_pres_hpa","obs_u_ms","obs_v_ms","obs_precip_mm","obs_rh_pct"]
SPATIAL_FEATURES = ["lat_norm", "lon_norm"]


def load_and_join() -> pd.DataFrame:
    gfs = pd.read_csv(RAW_DIR / "gfs_proxy.csv", parse_dates=["time"])
    obs = pd.read_csv(RAW_DIR / "metar_obs.csv",  parse_dates=["time"])

    # Round GFS to hour for alignment
    gfs["time"] = gfs["time"].dt.round("h")

    merged = gfs.merge(obs, on=["station", "time"], how="inner")
    print(f"Joined rows: {len(merged)}")
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    all_cols = GFS_FEATURES + OBS_FEATURES
    df = df.dropna(subset=all_cols)

    # Clip obvious outliers
    df = df[df["gfs_temp_c"].between(-60, 60)]
    df = df[df["obs_temp_c"].between(-60, 60)]
    df = df[df["gfs_pres_hpa"].between(880, 1080)]
    df = df[df["obs_pres_hpa"].between(880, 1080)]
    df = df[df["gfs_rh_pct"].between(0, 105)]
    df = df[df["obs_rh_pct"].between(0, 105)]
    df = df[df["gfs_precip_mm"] >= 0]
    df = df[df["obs_precip_mm"] >= 0]

    # Normalized spatial coordinates
    df["lat_norm"] = df["lat"] / 90.0
    df["lon_norm"] = df["lon"] / 180.0

    print(f"After cleaning: {len(df)} rows")
    return df.reset_index(drop=True)


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in GFS_FEATURES + OBS_FEATURES:
        stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std()) + 1e-8}
    return stats


def normalize(df: pd.DataFrame, stats: dict, cols: list) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for i, col in enumerate(cols):
        out[:, i] = (df[col].values - stats[col]["mean"]) / stats[col]["std"]
    return out


def build_tensors(df: pd.DataFrame, stats: dict):
    X_gfs     = normalize(df, stats, GFS_FEATURES)
    X_obs     = normalize(df, stats, OBS_FEATURES)
    X_spatial = df[SPATIAL_FEATURES].values.astype(np.float32)

    return (
        torch.from_numpy(X_gfs),
        torch.from_numpy(X_spatial),
        torch.from_numpy(X_obs),
    )


def main():
    df = load_and_join()
    df = clean(df)

    # Compute normalization stats on full dataset before splitting
    stats = compute_stats(df)
    torch.save(stats, PROC_DIR / "stats.pt")
    print("Saved normalization stats")

    # Shuffle and split 80/20
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:]

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        gfs, spatial, obs = build_tensors(split_df, stats)
        torch.save({"gfs": gfs, "spatial": spatial, "obs": obs},
                   PROC_DIR / f"{split_name}.pt")
        print(f"Saved {split_name}.pt  — {len(split_df)} samples")

    # Print bias summary (what the model needs to learn)
    print("\nMean GFS bias per variable (GFS − METAR):")
    for gfs_col, obs_col in zip(GFS_FEATURES, OBS_FEATURES):
        bias = (df[gfs_col] - df[obs_col]).mean()
        std  = (df[gfs_col] - df[obs_col]).std()
        print(f"  {gfs_col:20s}: bias={bias:+.3f}  std={std:.3f}")


if __name__ == "__main__":
    main()
