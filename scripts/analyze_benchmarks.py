"""
Analyzes benchmark CSV exported from the in-app benchmark harness.

Usage:
    python scripts/analyze_benchmarks.py benchmark_log.csv

Output:
    - Latency table: mean / P95 / P99 per variant
    - Cache hit rate (Variant B)
    - Compute savings estimate
    - Latency distribution plot (saved to benchmark_results.png)
"""

import sys
import csv
import statistics
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — skipping plot. pip install matplotlib")


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "variant": row["variant"],
                "inference_ms": int(row["inference_ms"]),
                "cache_hit": int(row["cache_hit"]),
                "timestamp": int(row["timestamp"]),
            })
    return rows


def percentile(data: list[float], p: float) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def analyze(rows: list[dict]) -> None:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    print("\n" + "=" * 60)
    print("  BAYESIAN WEATHER MOBILE — BENCHMARK RESULTS")
    print("=" * 60)

    summaries = {}
    for variant in sorted(by_variant.keys()):
        data = by_variant[variant]
        latencies = [r["inference_ms"] for r in data]
        hits = sum(r["cache_hit"] for r in data)
        total = len(data)

        mean_ms = statistics.mean(latencies)
        p95_ms = percentile(latencies, 95)
        p99_ms = percentile(latencies, 99)
        hit_rate = hits / total if total > 0 else 0.0

        summaries[variant] = {
            "latencies": latencies,
            "mean": mean_ms,
            "p95": p95_ms,
            "p99": p99_ms,
            "hit_rate": hit_rate,
            "n": total,
        }

        print(f"\n  Variant {variant}  (n={total})")
        print(f"    Mean latency : {mean_ms:.1f} ms")
        print(f"    P95  latency : {p95_ms:.1f} ms")
        print(f"    P99  latency : {p99_ms:.1f} ms")
        if variant == "B":
            print(f"    Cache hit rate: {hit_rate * 100:.1f}%")

    if "A" in summaries and "B" in summaries:
        a, b = summaries["A"], summaries["B"]
        speedup = a["mean"] / b["mean"] if b["mean"] > 0 else float("inf")
        savings = (1 - b["mean"] / a["mean"]) * 100 if a["mean"] > 0 else 0

        print("\n" + "-" * 60)
        print("  COMPARATIVE SUMMARY")
        print("-" * 60)
        print(f"    Latency speedup  (B vs A) : {speedup:.2f}×")
        print(f"    Compute savings  (B vs A) : {savings:.1f}%")
        print(f"    Cache hit rate   (B)      : {b['hit_rate'] * 100:.1f}%")
        mae_note = "Run evaluate.py to compute MAE improvement over raw GFS."
        print(f"\n  Note: {mae_note}")

    print("=" * 60 + "\n")

    if HAS_MPL and len(by_variant) > 0:
        _plot(summaries)


def _plot(summaries: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Benchmark Results: Variant A (GPU-always) vs Variant B (Cache-gated)", fontsize=13)

    colors = {"A": "#7c4dff", "B": "#00bfa5"}

    # Latency distribution (box plot)
    ax = axes[0]
    data_for_box = [summaries[v]["latencies"] for v in sorted(summaries.keys())]
    labels = [f"Variant {v}" for v in sorted(summaries.keys())]
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True, notch=False)
    for patch, variant in zip(bp["boxes"], sorted(summaries.keys())):
        patch.set_facecolor(colors.get(variant, "#888888"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Mean / P95 / P99 bar chart
    ax2 = axes[1]
    variants = sorted(summaries.keys())
    x = range(len(variants))
    width = 0.25
    metrics = [("Mean", "mean"), ("P95", "p95"), ("P99", "p99")]
    metric_colors = ["#1565c0", "#0288d1", "#29b6f6"]

    for i, (label, key) in enumerate(metrics):
        values = [summaries[v][key] for v in variants]
        ax2.bar([xi + i * width for xi in x], values, width, label=label,
                color=metric_colors[i], alpha=0.85)

    ax2.set_xticks([xi + width for xi in x])
    ax2.set_xticklabels([f"Variant {v}" for v in variants])
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Mean / P95 / P99 Latency")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path("benchmark_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved to {out_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmarks.py <benchmark_log.csv>")
        sys.exit(1)

    rows = load_csv(sys.argv[1])
    if not rows:
        print("No data found in CSV.")
        sys.exit(1)

    analyze(rows)
