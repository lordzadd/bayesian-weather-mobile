#!/usr/bin/env python3
"""
Research Spike — Compare LSTM vs PatchTST vs Bayesian Calibration

End-to-end pipeline:
  1. Prepare sequence data (if not cached)
  2. Train LSTM forecaster
  3. Train PatchTST forecaster
  4. Generate base model predictions
  5. Train Bayesian calibration on each base model
  6. Evaluate all models + baselines
  7. Print comparison table

Models tested:
  - Raw GFS          (baseline — no model, just NWP forecast)
  - Persistence      (baseline — last observation carried forward)
  - LSTM             (12h lookback sequential model)
  - PatchTST         (48h lookback patched transformer)
  - LSTM + BayCal    (LSTM with Bayesian calibration layer)
  - PatchTST + BayCal (PatchTST with Bayesian calibration layer)

Usage:
  python run_spike.py [--device cuda] [--epochs 100] [--skip-data]
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lstm_forecaster import LSTMForecaster
from patchtst_forecaster import PatchTSTForecaster
from bayesian_cal import BayesianCalibration
from train_utils import (
    train_base_model, train_cal_model, generate_predictions, count_params,
)
from evaluate_spike import evaluate_predictions, print_comparison, HORIZONS

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def prepare_if_needed():
    if (DATA_DIR / "train.pt").exists() and (DATA_DIR / "val.pt").exists():
        print("Data already prepared, skipping.")
        return
    print("Preparing sequence data...")
    import sys as _sys
    _data_dir = str(Path(__file__).parent.parent / "data")
    if _data_dir not in _sys.path:
        _sys.path.insert(0, _data_dir)
    import prepare_sequences
    prepare_sequences.main()


def load_data(device="cpu"):
    train = torch.load(DATA_DIR / "train.pt", weights_only=False)
    val = torch.load(DATA_DIR / "val.pt", weights_only=False)
    stats = torch.load(DATA_DIR / "stats.pt", weights_only=False)
    print(f"Train: {train['obs_hist'].shape[0]} windows  |  Val: {val['obs_hist'].shape[0]} windows")
    return train, val, stats


def eval_baselines(val, stats):
    """Evaluate no-model baselines."""
    results = {}

    # Raw GFS: use GFS forecast as prediction (no uncertainty)
    results["Raw GFS"] = evaluate_predictions(
        val["gfs_targets"], None, val["obs_targets"], stats
    )

    # Persistence: last observation repeated at all horizons
    last_obs = val["obs_hist"][:, -1:, :].expand_as(val["obs_targets"])
    results["Persistence"] = evaluate_predictions(
        last_obs, None, val["obs_targets"], stats
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")

    # --- 1. Data ---
    if not args.skip_data:
        prepare_if_needed()
    train, val, stats = load_data()

    # --- 2. Baselines ---
    print("\n" + "=" * 60)
    print("Evaluating baselines...")
    print("=" * 60)
    all_results = eval_baselines(val, stats)
    for name, r in all_results.items():
        print(f"  {name}: MAE={r['mae_overall']:.4f}")

    # --- 3. Train LSTM (with GFS in prediction head — existing approach) ---
    lstm = LSTMForecaster(
        n_vars=6, hidden_dim=128, n_layers=2,
        n_horizons=len(HORIZONS), lstm_lookback=12, include_gfs=True,
    )
    lstm = train_base_model(
        lstm, train, val, name="lstm",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 4. Train Pure LSTM (obs-only, no GFS — for proper fusion) ---
    lstm_pure = LSTMForecaster(
        n_vars=6, hidden_dim=128, n_layers=2,
        n_horizons=len(HORIZONS), lstm_lookback=12, include_gfs=False,
    )
    lstm_pure = train_base_model(
        lstm_pure, train, val, name="lstm_pure",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 5. Train PatchTST ---
    patchtst = PatchTSTForecaster(
        n_vars=6, seq_len=48, patch_len=6,
        d_model=64, n_heads=4, n_layers=2,
        n_horizons=len(HORIZONS),
    )
    patchtst = train_base_model(
        patchtst, train, val, name="patchtst",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 6. Generate base predictions for calibration ---
    print("\nGenerating base model predictions for calibration training...")
    lstm_mu_tr, lstm_lv_tr = generate_predictions(lstm, train, args.device)
    lstm_mu_val, lstm_lv_val = generate_predictions(lstm, val, args.device)
    pure_mu_tr, pure_lv_tr = generate_predictions(lstm_pure, train, args.device)
    pure_mu_val, pure_lv_val = generate_predictions(lstm_pure, val, args.device)
    ptst_mu_tr, ptst_lv_tr = generate_predictions(patchtst, train, args.device)
    ptst_mu_val, ptst_lv_val = generate_predictions(patchtst, val, args.device)
    print("  Done.")

    # --- 7. Train BayCal on LSTM (already has GFS baked in) ---
    lstm_cal = BayesianCalibration(n_vars=6, hidden_dim=64, n_horizons=len(HORIZONS))
    lstm_cal = train_cal_model(
        lstm_cal,
        lstm_mu_tr, lstm_lv_tr, train,
        lstm_mu_val, lstm_lv_val, val,
        name="lstm_baycal",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 8. Train BayCal on Pure LSTM (true NOAA fusion — the arch doc design) ---
    pure_cal = BayesianCalibration(n_vars=6, hidden_dim=64, n_horizons=len(HORIZONS))
    pure_cal = train_cal_model(
        pure_cal,
        pure_mu_tr, pure_lv_tr, train,
        pure_mu_val, pure_lv_val, val,
        name="pure_lstm_baycal",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 9. Train BayCal on PatchTST ---
    ptst_cal = BayesianCalibration(n_vars=6, hidden_dim=64, n_horizons=len(HORIZONS))
    ptst_cal = train_cal_model(
        ptst_cal,
        ptst_mu_tr, ptst_lv_tr, train,
        ptst_mu_val, ptst_lv_val, val,
        name="patchtst_baycal",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # --- 10. Evaluate all models on val ---
    print("\n" + "=" * 60)
    print("Evaluating all models on validation set...")
    print("=" * 60)

    # LSTM (has GFS in head)
    all_results["LSTM"] = evaluate_predictions(
        lstm_mu_val, lstm_lv_val, val["obs_targets"], stats
    )

    # Pure LSTM (obs-only, no GFS)
    all_results["LSTM (pure)"] = evaluate_predictions(
        pure_mu_val, pure_lv_val, val["obs_targets"], stats
    )

    # PatchTST
    all_results["PatchTST"] = evaluate_predictions(
        ptst_mu_val, ptst_lv_val, val["obs_targets"], stats
    )

    # LSTM + BayCal (GFS seen twice)
    lstm_cal.eval()
    with torch.no_grad():
        cal_mu, cal_lv = lstm_cal(
            lstm_mu_val, lstm_lv_val,
            val["gfs_targets"], val["spatial"], val["temporal"],
        )
    all_results["LSTM + BayCal"] = evaluate_predictions(
        cal_mu, cal_lv, val["obs_targets"], stats
    )

    # Pure LSTM + BayCal (true NOAA fusion)
    pure_cal.eval()
    with torch.no_grad():
        cal_mu, cal_lv = pure_cal(
            pure_mu_val, pure_lv_val,
            val["gfs_targets"], val["spatial"], val["temporal"],
        )
    all_results["Pure LSTM + Fusion"] = evaluate_predictions(
        cal_mu, cal_lv, val["obs_targets"], stats
    )

    # PatchTST + BayCal
    ptst_cal.eval()
    with torch.no_grad():
        cal_mu, cal_lv = ptst_cal(
            ptst_mu_val, ptst_lv_val,
            val["gfs_targets"], val["spatial"], val["temporal"],
        )
    all_results["PatchTST + BayCal"] = evaluate_predictions(
        cal_mu, cal_lv, val["obs_targets"], stats
    )

    # --- 11. Print comparison ---
    model_names = [
        "Raw GFS", "Persistence",
        "LSTM", "LSTM (pure)", "PatchTST",
        "LSTM + BayCal", "Pure LSTM + Fusion", "PatchTST + BayCal",
    ]
    print_comparison(all_results, model_names)

    # --- 12. Summary ---
    print(f"\n{'='*80}")
    print("MODEL SIZES")
    print(f"{'='*80}")
    print(f"  {'LSTM (w/ GFS)':<25} {count_params(lstm):>10,} params")
    print(f"  {'LSTM (pure, no GFS)':<25} {count_params(lstm_pure):>10,} params")
    print(f"  {'PatchTST':<25} {count_params(patchtst):>10,} params")
    print(f"  {'BayCal (each)':<25} {count_params(lstm_cal):>10,} params")
    print(f"  {'Pure LSTM + Fusion':<25} {count_params(lstm_pure) + count_params(pure_cal):>10,} params")

    # Best model recommendation
    candidates = ["LSTM", "LSTM (pure)", "PatchTST",
                   "LSTM + BayCal", "Pure LSTM + Fusion", "PatchTST + BayCal"]
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    best_name = min(candidates, key=lambda n: all_results[n]["mae_overall"])
    best_mae = all_results[best_name]["mae_overall"]
    gfs_mae = all_results["Raw GFS"]["mae_overall"]
    improvement = (1 - best_mae / gfs_mae) * 100

    print(f"  Best overall: {best_name}")
    print(f"  MAE: {best_mae:.4f} ({improvement:+.1f}% vs Raw GFS)")

    # Per-horizon winner
    print("\n  Per-horizon winners:")
    for h in HORIZONS:
        winner = min(candidates, key=lambda n: all_results[n][f"mae_h{h}"])
        mae = all_results[winner][f"mae_h{h}"]
        print(f"    h+{h:2d}: {winner:<25} MAE={mae:.4f}")

    # --- 13. Blend weight analysis ---
    print(f"\n{'='*80}")
    print("BLEND WEIGHTS (alpha: 1.0 = trust base model, 0.0 = trust GFS)")
    print(f"{'='*80}")
    var_names = ["Temp", "Pres", "U", "V", "Precip", "RH"]
    for cal, base_mu, base_lv, label in [
        (lstm_cal, lstm_mu_val, lstm_lv_val, "LSTM + BayCal (GFS seen twice)"),
        (pure_cal, pure_mu_val, pure_lv_val, "Pure LSTM + Fusion (true fusion)"),
        (ptst_cal, ptst_mu_val, ptst_lv_val, "PatchTST + BayCal"),
    ]:
        cal.eval()
        with torch.no_grad():
            alphas = cal.get_blend_weights(
                base_mu, base_lv,
                val["gfs_targets"], val["spatial"], val["temporal"],
            )  # (N, H, V)
        mean_alpha = alphas.mean(dim=0)  # (H, V)
        print(f"\n  {label}:")
        header = f"    {'Horizon':<10}" + "".join(f"{v:>8}" for v in var_names)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for i, h in enumerate(HORIZONS):
            row = f"    h+{h:<7}"
            for j in range(len(var_names)):
                row += f"{mean_alpha[i, j]:.3f}   "
            print(row)

    print()


if __name__ == "__main__":
    main()
