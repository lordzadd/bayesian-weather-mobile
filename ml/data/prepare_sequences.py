"""
Build sequence datasets for the research spike.

Creates sliding windows from raw GFS + observation CSVs with multi-horizon
targets at +1h, +3h, +6h, +12h, +24h. Uses stride=6 to keep dataset
manageable while preserving diversity.

Input:  raw/{gfs_proxy.csv, metar_obs.csv}
Output: processed/{train.pt, val.pt, stats.pt}

Each .pt file contains:
  obs_hist    (N, 48, 6)  — normalized observation history
  gfs_targets (N, 5, 6)   — normalized GFS forecast at each horizon
  spatial     (N, 2)       — lat/lon normalized
  temporal    (N, 5, 4)    — sin/cos hour+doy at each target time
  obs_targets (N, 5, 6)   — normalized observation truth at each horizon
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"

GFS_COLS = ["gfs_temp_c", "gfs_pres_hpa", "gfs_u_ms", "gfs_v_ms", "gfs_precip_mm", "gfs_rh_pct"]
OBS_COLS = ["obs_temp_c", "obs_pres_hpa", "obs_u_ms", "obs_v_ms", "obs_precip_mm", "obs_rh_pct"]
VAR_NAMES = ["temp", "pressure", "u_wind", "v_wind", "precip", "humidity"]

LOOKBACK = 48
HORIZONS = [1, 3, 6, 12, 24]
STRIDE = 6
VAL_START = pd.Timestamp("2024-07-01")


def load_and_merge() -> pd.DataFrame:
    gfs = pd.read_csv(RAW_DIR / "gfs_proxy.csv", parse_dates=["time"])
    obs = pd.read_csv(RAW_DIR / "metar_obs.csv", parse_dates=["time"])
    gfs["time"] = gfs["time"].dt.round("h")
    df = gfs.merge(obs, on=["station", "time"], how="inner")
    df = df.dropna(subset=GFS_COLS + OBS_COLS)

    # Compute temporal features if not present in the merged data
    temporal_cols = ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    if not all(c in df.columns for c in temporal_cols):
        hours = df["time"].dt.hour + df["time"].dt.minute / 60.0
        doy = df["time"].dt.dayofyear
        df["sin_hour"] = np.sin(hours * 2 * np.pi / 24)
        df["cos_hour"] = np.cos(hours * 2 * np.pi / 24)
        df["sin_doy"] = np.sin(doy * 2 * np.pi / 365.25)
        df["cos_doy"] = np.cos(doy * 2 * np.pi / 365.25)

    df = df.sort_values(["station", "time"]).reset_index(drop=True)
    print(f"Merged: {len(df)} rows, {df['station'].nunique()} stations")
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in GFS_COLS + OBS_COLS:
        stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std()) + 1e-8}
    return stats


def normalize_array(df: pd.DataFrame, stats: dict, cols: list) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for i, col in enumerate(cols):
        out[:, i] = (df[col].values - stats[col]["mean"]) / stats[col]["std"]
    return out


def build_windows(df: pd.DataFrame, stats: dict) -> dict:
    max_h = max(HORIZONS)
    total_needed = LOOKBACK + max_h

    gfs_n = normalize_array(df, stats, GFS_COLS)
    obs_n = normalize_array(df, stats, OBS_COLS)
    spatial = np.column_stack([
        df["lat"].values / 90.0,
        df["lon"].values / 180.0,
    ]).astype(np.float32)
    temporal = df[["sin_hour", "cos_hour", "sin_doy", "cos_doy"]].values.astype(np.float32)
    times = df["time"].values
    stations = df["station"].values

    buffers = {k: [] for k in ["obs_hist", "gfs_targets", "spatial", "temporal", "obs_targets"]}

    for station in df["station"].unique():
        idx = np.where(stations == station)[0]
        if len(idx) < total_needed:
            continue

        st_times = times[idx]
        diffs = np.diff(st_times) / np.timedelta64(1, "h")
        breaks = np.where(diffs != 1.0)[0]
        seg_starts = np.concatenate([[0], breaks + 1])
        seg_ends = np.concatenate([breaks + 1, [len(idx)]])

        for s, e in zip(seg_starts, seg_ends):
            seg_len = e - s
            if seg_len < total_needed:
                continue

            seg = idx[s:e]
            for i in range(0, seg_len - total_needed + 1, STRIDE):
                hist = seg[i : i + LOOKBACK]
                buffers["obs_hist"].append(obs_n[hist])
                buffers["spatial"].append(spatial[seg[i]])

                tgt_gfs, tgt_obs, tgt_tmp = [], [], []
                for h in HORIZONS:
                    t = seg[i + LOOKBACK + h - 1]
                    tgt_gfs.append(gfs_n[t])
                    tgt_obs.append(obs_n[t])
                    tgt_tmp.append(temporal[t])

                buffers["gfs_targets"].append(np.stack(tgt_gfs))
                buffers["obs_targets"].append(np.stack(tgt_obs))
                buffers["temporal"].append(np.stack(tgt_tmp))

    return {k: torch.from_numpy(np.stack(v)) for k, v in buffers.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and merging raw data...")
    df = load_and_merge()

    train_df = df[df["time"] < VAL_START].reset_index(drop=True)
    val_df = df[df["time"] >= VAL_START].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows  |  Val: {len(val_df)} rows")

    stats = compute_stats(train_df)
    torch.save(stats, OUT_DIR / "stats.pt")
    print("Saved stats.pt")

    for name, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\nBuilding {name} windows (lookback={LOOKBACK}, stride={STRIDE})...")
        data = build_windows(split_df, stats)
        torch.save(data, OUT_DIR / f"{name}.pt")
        n = data["obs_hist"].shape[0]
        print(f"  {name}: {n} windows  |  obs_hist {tuple(data['obs_hist'].shape)}")
        print(f"  gfs_targets {tuple(data['gfs_targets'].shape)}  |  obs_targets {tuple(data['obs_targets'].shape)}")


if __name__ == "__main__":
    main()
