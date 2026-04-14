"""
Evaluation metrics for all models.

Reports MAE in physical units (denormalized) and calibration coverage.
Produces per-horizon and per-variable breakdowns for the comparison table.
"""

import torch
import numpy as np

# Must match prepare_data.py
HORIZONS = [1, 3, 6, 12, 24]
OBS_COLS = ["obs_temp_c", "obs_pres_hpa", "obs_u_ms", "obs_v_ms", "obs_precip_mm", "obs_rh_pct"]
VAR_NAMES = ["Temp(°C)", "Pres(hPa)", "U(m/s)", "V(m/s)", "Precip(mm)", "RH(%)"]


def _get_obs_stds(stats):
    """Extract per-variable std from stats dict for denormalization."""
    return torch.tensor([stats[c]["std"] for c in OBS_COLS])


def evaluate_predictions(mu, logvar, targets, stats):
    """
    Evaluate predictions against targets.

    Args:
        mu:      (N, H, V) predicted mean in normalized space
        logvar:  (N, H, V) predicted log-variance (None for baselines)
        targets: (N, H, V) observation truth in normalized space
        stats:   normalization stats dict

    Returns:
        dict with per-horizon MAE (physical), calibration, and summary metrics
    """
    stds = _get_obs_stds(stats)  # (V,)
    errors_norm = (mu - targets).abs()
    errors_phys = errors_norm * stds.unsqueeze(0).unsqueeze(0)  # denormalize

    results = {}

    # Per-horizon metrics
    for i, h in enumerate(HORIZONS):
        results[f"mae_h{h}"] = errors_phys[:, i].mean().item()
        results[f"rmse_h{h}"] = (
            ((mu[:, i] - targets[:, i]) ** 2).mean().sqrt() * stds.mean()
        ).item()

        if logvar is not None:
            sigma = (logvar[:, i] / 2).exp()
            resid = (targets[:, i] - mu[:, i]).abs()
            results[f"cov1_h{h}"] = (resid <= sigma).float().mean().item()
            results[f"cov2_h{h}"] = (resid <= 2 * sigma).float().mean().item()
            results[f"mean_sigma_h{h}"] = (sigma * stds.unsqueeze(0)).mean().item()

    # Per-variable metrics (averaged over horizons)
    for j, var in enumerate(VAR_NAMES):
        results[f"mae_{var}"] = errors_phys[:, :, j].mean().item()

    # Overall
    results["mae_overall"] = errors_phys.mean().item()

    return results


def print_comparison(all_results, model_names):
    """Print a formatted comparison table."""

    # --- Per-horizon MAE ---
    print(f"\n{'='*80}")
    print("PER-HORIZON MAE (physical units, averaged over variables)")
    print(f"{'='*80}")
    header = f"{'Model':<25}" + "".join(f"{'h+' + str(h):>10}" for h in HORIZONS) + f"{'Overall':>10}"
    print(header)
    print("-" * len(header))
    for name in model_names:
        r = all_results[name]
        row = f"{name:<25}"
        for h in HORIZONS:
            row += f"{r.get(f'mae_h{h}', float('nan')):>10.3f}"
        row += f"{r['mae_overall']:>10.3f}"
        print(row)

    # --- Calibration coverage ---
    has_cal = any(f"cov1_h1" in all_results[n] for n in model_names)
    if has_cal:
        print(f"\n{'='*80}")
        print("1σ COVERAGE (expect ~68%)  |  2σ COVERAGE (expect ~95%)")
        print(f"{'='*80}")
        header = f"{'Model':<25}" + "".join(f"{'h+' + str(h):>10}" for h in HORIZONS)
        print("1σ coverage:")
        print(header)
        print("-" * len(header))
        for name in model_names:
            r = all_results[name]
            if f"cov1_h1" not in r:
                continue
            row = f"{name:<25}"
            for h in HORIZONS:
                row += f"{r[f'cov1_h{h}'] * 100:>9.1f}%"
            print(row)

        print("\n2σ coverage:")
        print(header)
        print("-" * len(header))
        for name in model_names:
            r = all_results[name]
            if f"cov2_h1" not in r:
                continue
            row = f"{name:<25}"
            for h in HORIZONS:
                row += f"{r[f'cov2_h{h}'] * 100:>9.1f}%"
            print(row)

    # --- Per-variable MAE ---
    print(f"\n{'='*80}")
    print("PER-VARIABLE MAE (averaged over horizons)")
    print(f"{'='*80}")
    header = f"{'Model':<25}" + "".join(f"{v:>12}" for v in VAR_NAMES)
    print(header)
    print("-" * len(header))
    for name in model_names:
        r = all_results[name]
        row = f"{name:<25}"
        for v in VAR_NAMES:
            row += f"{r[f'mae_{v}']:>12.3f}"
        print(row)

    # --- Mean predicted sigma (for calibrated models) ---
    if has_cal:
        print(f"\n{'='*80}")
        print("MEAN PREDICTED σ (physical units) — lower is tighter bounds")
        print(f"{'='*80}")
        header = f"{'Model':<25}" + "".join(f"{'h+' + str(h):>10}" for h in HORIZONS)
        print(header)
        print("-" * len(header))
        for name in model_names:
            r = all_results[name]
            if f"mean_sigma_h1" not in r:
                continue
            row = f"{name:<25}"
            for h in HORIZONS:
                row += f"{r[f'mean_sigma_h{h}']:>10.3f}"
            print(row)
