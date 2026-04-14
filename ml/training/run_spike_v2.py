#!/usr/bin/env python3
"""
Research Spike v2 — Teacher distillation + per-horizon BayCal

Builds on v1 findings (Pure LSTM + Fusion won). Now tests:
  1. Larger teacher LSTM (256 hidden, 3 layers) for richer dynamics
  2. Knowledge distillation from teacher to student-sized LSTM
  3. Per-horizon BayCal (dedicated fusion model per forecast hour)
  4. Combined: distilled student + per-horizon fusion

Focus: can we improve h+3 where the LSTM's sequential skill is fading?

Usage:
  python run_spike_v2.py [--device cuda] [--epochs 100]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lstm_forecaster import LSTMForecaster
from bayesian_cal import BayesianCalibration, PerHorizonBayCal
from train_utils import (
    train_base_model, train_cal_model, train_distilled,
    generate_predictions, count_params,
)
from evaluate_spike import evaluate_predictions, print_comparison, HORIZONS

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    train = torch.load(DATA_DIR / "train.pt", weights_only=False)
    val = torch.load(DATA_DIR / "val.pt", weights_only=False)
    stats = torch.load(DATA_DIR / "stats.pt", weights_only=False)
    print(f"Train: {train['obs_hist'].shape[0]} windows  |  Val: {val['obs_hist'].shape[0]} windows")
    return train, val, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    train, val, stats = load_data()
    NH = len(HORIZONS)

    all_results = {}

    # ================================================================
    # 1. Train the v1 baseline: pure student LSTM (no GFS, 128h/2L)
    # ================================================================
    student_base = LSTMForecaster(
        n_vars=6, hidden_dim=128, n_layers=2,
        n_horizons=NH, lstm_lookback=12, include_gfs=False,
    )
    student_base = train_base_model(
        student_base, train, val, name="v2_student_base",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 2. Train the teacher: larger pure LSTM (256h/3L)
    # ================================================================
    teacher = LSTMForecaster(
        n_vars=6, hidden_dim=256, n_layers=3,
        n_horizons=NH, lstm_lookback=12, include_gfs=False,
    )
    teacher = train_base_model(
        teacher, train, val, name="v2_teacher",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 3. Generate teacher predictions for distillation
    # ================================================================
    print("\nGenerating teacher predictions...")
    t_mu_tr, t_lv_tr = generate_predictions(teacher, train, args.device)
    t_mu_val, t_lv_val = generate_predictions(teacher, val, args.device)

    # ================================================================
    # 4. Distill teacher → student
    # ================================================================
    student_distilled = LSTMForecaster(
        n_vars=6, hidden_dim=128, n_layers=2,
        n_horizons=NH, lstm_lookback=12, include_gfs=False,
    )
    student_distilled = train_distilled(
        student_distilled,
        t_mu_tr, t_lv_tr, t_mu_val, t_lv_val,
        train, val, name="v2_student_distilled",
        alpha=0.5, beta=0.3,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 5. Generate base predictions for BayCal training
    # ================================================================
    print("\nGenerating base model predictions...")
    base_mu_tr, base_lv_tr = generate_predictions(student_base, train, args.device)
    base_mu_val, base_lv_val = generate_predictions(student_base, val, args.device)
    dist_mu_tr, dist_lv_tr = generate_predictions(student_distilled, train, args.device)
    dist_mu_val, dist_lv_val = generate_predictions(student_distilled, val, args.device)

    # ================================================================
    # 6. Train shared BayCal on base student (v1 approach, for reference)
    # ================================================================
    shared_cal = BayesianCalibration(n_vars=6, hidden_dim=64, n_horizons=NH)
    shared_cal = train_cal_model(
        shared_cal, base_mu_tr, base_lv_tr, train,
        base_mu_val, base_lv_val, val,
        name="v2_shared_cal",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 7. Train per-horizon BayCal on base student
    # ================================================================
    perhz_cal = PerHorizonBayCal(n_vars=6, hidden_dim=64, n_horizons=NH)
    perhz_cal = train_cal_model(
        perhz_cal, base_mu_tr, base_lv_tr, train,
        base_mu_val, base_lv_val, val,
        name="v2_perhz_cal",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 8. Train per-horizon BayCal on distilled student (the full pipeline)
    # ================================================================
    perhz_cal_dist = PerHorizonBayCal(n_vars=6, hidden_dim=64, n_horizons=NH)
    perhz_cal_dist = train_cal_model(
        perhz_cal_dist, dist_mu_tr, dist_lv_tr, train,
        dist_mu_val, dist_lv_val, val,
        name="v2_perhz_cal_dist",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, device=args.device,
    )

    # ================================================================
    # 9. Evaluate everything
    # ================================================================
    print("\n" + "=" * 60)
    print("Evaluating all models...")
    print("=" * 60)

    # Raw GFS baseline
    all_results["Raw GFS"] = evaluate_predictions(
        val["gfs_targets"], None, val["obs_targets"], stats
    )

    # Teacher standalone
    all_results["Teacher (256h/3L)"] = evaluate_predictions(
        t_mu_val, t_lv_val, val["obs_targets"], stats
    )

    # Base student standalone
    all_results["Student (base)"] = evaluate_predictions(
        base_mu_val, base_lv_val, val["obs_targets"], stats
    )

    # Distilled student standalone
    all_results["Student (distilled)"] = evaluate_predictions(
        dist_mu_val, dist_lv_val, val["obs_targets"], stats
    )

    # v1 approach: base student + shared BayCal
    shared_cal.eval()
    with torch.no_grad():
        mu, lv = shared_cal(base_mu_val, base_lv_val,
                            val["gfs_targets"], val["spatial"], val["temporal"])
    all_results["v1: Student + SharedCal"] = evaluate_predictions(
        mu, lv, val["obs_targets"], stats
    )

    # Per-horizon BayCal on base student
    perhz_cal.eval()
    with torch.no_grad():
        mu, lv = perhz_cal(base_mu_val, base_lv_val,
                           val["gfs_targets"], val["spatial"], val["temporal"])
    all_results["Student + PerHzCal"] = evaluate_predictions(
        mu, lv, val["obs_targets"], stats
    )

    # Per-horizon BayCal on distilled student (full pipeline)
    perhz_cal_dist.eval()
    with torch.no_grad():
        mu, lv = perhz_cal_dist(dist_mu_val, dist_lv_val,
                                val["gfs_targets"], val["spatial"], val["temporal"])
    all_results["Distilled + PerHzCal"] = evaluate_predictions(
        mu, lv, val["obs_targets"], stats
    )

    # ================================================================
    # 10. Print comparison
    # ================================================================
    model_names = [
        "Raw GFS",
        "Teacher (256h/3L)", "Student (base)", "Student (distilled)",
        "v1: Student + SharedCal",
        "Student + PerHzCal", "Distilled + PerHzCal",
    ]
    print_comparison(all_results, model_names)

    # Model sizes
    print(f"\n{'='*80}")
    print("MODEL SIZES")
    print(f"{'='*80}")
    print(f"  {'Teacher (256h/3L)':<30} {count_params(teacher):>10,} params")
    print(f"  {'Student (128h/2L)':<30} {count_params(student_base):>10,} params")
    print(f"  {'Shared BayCal':<30} {count_params(shared_cal):>10,} params")
    print(f"  {'PerHorizon BayCal':<30} {count_params(perhz_cal):>10,} params")
    print(f"  {'Student + SharedCal':<30} {count_params(student_base) + count_params(shared_cal):>10,} params")
    print(f"  {'Student + PerHzCal':<30} {count_params(student_base) + count_params(perhz_cal):>10,} params")
    print(f"  {'Distilled + PerHzCal':<30} {count_params(student_distilled) + count_params(perhz_cal_dist):>10,} params")

    # h+3 focus
    print(f"\n{'='*80}")
    print("h+3 FOCUS (the target horizon for improvement)")
    print(f"{'='*80}")
    for name in model_names:
        r = all_results[name]
        cov = f"  1σ={r.get('cov1_h3', 0)*100:.1f}%" if 'cov1_h3' in r else ""
        print(f"  {name:<30} MAE={r['mae_h3']:.4f}{cov}")

    # Best overall
    candidates = [n for n in model_names if n != "Raw GFS"]
    best = min(candidates, key=lambda n: all_results[n]["mae_overall"])
    best3 = min(candidates, key=lambda n: all_results[n]["mae_h3"])
    gfs_mae = all_results["Raw GFS"]["mae_overall"]
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"  Best overall: {best}  MAE={all_results[best]['mae_overall']:.4f}  ({(1 - all_results[best]['mae_overall']/gfs_mae)*100:+.1f}% vs GFS)")
    print(f"  Best at h+3:  {best3}  MAE={all_results[best3]['mae_h3']:.4f}")

    # Per-horizon winners
    print("\n  Per-horizon winners:")
    for h in HORIZONS:
        winner = min(candidates, key=lambda n: all_results[n][f"mae_h{h}"])
        mae = all_results[winner][f"mae_h{h}"]
        print(f"    h+{h:2d}: {winner:<30} MAE={mae:.4f}")

    # Blend weights for the full pipeline
    print(f"\n{'='*80}")
    print("BLEND WEIGHTS — Distilled + PerHzCal (full pipeline)")
    print(f"{'='*80}")
    var_names = ["Temp", "Pres", "U", "V", "Precip", "RH"]
    perhz_cal_dist.eval()
    with torch.no_grad():
        alphas = perhz_cal_dist.get_blend_weights(
            dist_mu_val, dist_lv_val,
            val["gfs_targets"], val["spatial"], val["temporal"],
        )
    mean_alpha = alphas.mean(dim=0)
    header = f"  {'Horizon':<10}" + "".join(f"{v:>8}" for v in var_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, h in enumerate(HORIZONS):
        row = f"  h+{h:<7}"
        for j in range(len(var_names)):
            row += f"{mean_alpha[i, j]:.3f}   "
        print(row)

    print()


if __name__ == "__main__":
    main()
