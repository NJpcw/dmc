"""Shared four-method multi-seed statistics and exports."""

from __future__ import annotations

import csv

import numpy as np


METHOD_LABELS = [
    "Current-state Raw-NGRC (k=1)",
    "Delay-embedded Raw-NGRC (k=2, s=1)",
    "BLI-NGRC (k=1)",
    "Delay-embedded BLI-NGRC (k=2, s=1)",
]


def relative_rmse_reduction(baseline, candidate):
    """Return a fractional reduction; zero/nonfinite baselines are undefined."""
    if not np.isfinite(baseline) or baseline <= 0.0 or not np.isfinite(candidate):
        return float("nan")
    return float(1.0 - candidate / baseline)


def rmse_statistics(values):
    values = np.asarray(values, dtype=float)
    finite = bool(np.all(np.isfinite(values)))
    mean = float(np.mean(values)) if finite else float("inf")
    median = float(np.median(values)) if finite else float("inf")
    standard_deviation = (
        float(np.std(values, ddof=1)) if finite and len(values) > 1 else 0.0
    )
    half_width = 1.96 * standard_deviation / np.sqrt(len(values))
    return {
        "mean_rmse": mean,
        "median_rmse": median,
        "rmse_std": standard_deviation,
        "mean_95_percent_ci": [float(mean - half_width), float(mean + half_width)],
    }


def pairwise_bli_summary(baseline, bli):
    baseline = np.asarray(baseline, dtype=float)
    bli = np.asarray(bli, dtype=float)
    finite = bool(np.all(np.isfinite(baseline)) and np.all(np.isfinite(bli)))
    mean_baseline = float(np.mean(baseline)) if finite else float("inf")
    mean_bli = float(np.mean(bli)) if finite else float("inf")
    reduction = relative_rmse_reduction(mean_baseline, mean_bli)
    win_count = int(np.sum(bli < baseline)) if finite else 0
    return {
        "relative_rmse_reduction": reduction,
        "win_count": win_count,
        "loss_count": int(len(baseline) - win_count),
    }


def summarize_seed_records(records):
    if not records:
        raise ValueError("At least one seed record is required.")
    raw = np.asarray([record["raw_rmse"] for record in records], dtype=float)
    raw_delay = np.asarray([record['raw_delay_rmse'] for record in records], dtype=float)
    bli = np.asarray([record["bli_rmse"] for record in records], dtype=float)
    bli_delay = np.asarray([record['bli_delay_rmse'] for record in records], dtype=float)
    raw_statistics = rmse_statistics(raw)
    raw_delay_statistics = rmse_statistics(raw_delay)
    bli_statistics = rmse_statistics(bli)
    bli_delay_statistics = rmse_statistics(bli_delay)
    for method, statistics in (
        ("raw", raw_statistics), ("raw_delay", raw_delay_statistics),
        ("bli", bli_statistics), ("bli_delay", bli_delay_statistics),
    ):
        statistics["mean_failure_rate"] = float(np.mean(
            [record[f"{method}_failure_rate"] for record in records]
        ))
    versus_raw = pairwise_bli_summary(raw, bli)
    versus_raw_delay = pairwise_bli_summary(raw_delay, bli)
    bli_delay_versus_raw = pairwise_bli_summary(raw, bli_delay)
    bli_delay_versus_raw_delay = pairwise_bli_summary(raw_delay, bli_delay)
    bli_delay_versus_bli = pairwise_bli_summary(bli, bli_delay)
    reduction = versus_raw["relative_rmse_reduction"]
    win_count = versus_raw["win_count"]
    return {
        "num_seeds": len(records),
        "method_statistics": {
            "raw": raw_statistics,
            "raw_delay": raw_delay_statistics,
            "bli": bli_statistics,
            "bli_delay": bli_delay_statistics,
        },
        "mean_raw_rmse": raw_statistics["mean_rmse"],
        "mean_raw_delay_rmse": raw_delay_statistics["mean_rmse"],
        "mean_bli_rmse": bli_statistics["mean_rmse"],
        "mean_bli_delay_rmse": bli_delay_statistics["mean_rmse"],
        "median_raw_rmse": raw_statistics["median_rmse"],
        "median_raw_delay_rmse": raw_delay_statistics["median_rmse"],
        "median_bli_rmse": bli_statistics["median_rmse"],
        "median_bli_delay_rmse": bli_delay_statistics["median_rmse"],
        "raw_rmse_std": raw_statistics["rmse_std"],
        "raw_delay_rmse_std": raw_delay_statistics["rmse_std"],
        "bli_rmse_std": bli_statistics["rmse_std"],
        "bli_delay_rmse_std": bli_delay_statistics["rmse_std"],
        "raw_mean_95_percent_ci": raw_statistics["mean_95_percent_ci"],
        "raw_delay_mean_95_percent_ci": raw_delay_statistics[
            "mean_95_percent_ci"
        ],
        "bli_mean_95_percent_ci": bli_statistics["mean_95_percent_ci"],
        "bli_delay_mean_95_percent_ci": bli_delay_statistics[
            "mean_95_percent_ci"
        ],
        "relative_rmse_reduction": reduction,
        "win_count": win_count,
        "loss_count": int(len(records) - win_count),
        "bli_vs_raw_current": versus_raw,
        "bli_vs_raw_delay": versus_raw_delay,
        "bli_delay_vs_raw_current": bli_delay_versus_raw,
        "bli_delay_vs_raw_delay": bli_delay_versus_raw_delay,
        "bli_delay_vs_bli": bli_delay_versus_bli,
    }


def build_seed_record(seed, results, elapsed_seconds):
    """Build one auditable four-method NGRC ablation record."""
    raw = results["raw"]
    raw_delay = results["raw_delay"]
    bli = results["bli"]
    bli_delay = results["bli_delay"]
    raw_stat = raw["external_stat"]
    raw_delay_stat = raw_delay["external_stat"]
    bli_stat = bli["external_stat"]
    bli_delay_stat = bli_delay["external_stat"]
    raw_rmse = float(raw_stat["finite_mean"])
    raw_delay_rmse = float(raw_delay_stat["finite_mean"])
    bli_rmse = float(bli_stat["finite_mean"])
    bli_delay_rmse = float(bli_delay_stat["finite_mean"])
    reduction_vs_raw = relative_rmse_reduction(raw_rmse, bli_rmse)
    reduction_vs_raw_delay = relative_rmse_reduction(raw_delay_rmse, bli_rmse)
    bli_delay_vs_raw = relative_rmse_reduction(raw_rmse, bli_delay_rmse)
    bli_delay_vs_raw_delay = relative_rmse_reduction(raw_delay_rmse, bli_delay_rmse)
    bli_delay_vs_bli = relative_rmse_reduction(bli_rmse, bli_delay_rmse)
    return {
        "seed": int(seed),
        "raw_rmse": raw_rmse,
        "raw_delay_rmse": raw_delay_rmse,
        "bli_rmse": bli_rmse,
        "bli_delay_rmse": bli_delay_rmse,
        "relative_reduction": reduction_vs_raw,
        "relative_reduction_vs_raw_current": reduction_vs_raw,
        "relative_reduction_vs_raw_delay": reduction_vs_raw_delay,
        "relative_reduction_bli_delay_vs_raw_current": bli_delay_vs_raw,
        "relative_reduction_bli_delay_vs_raw_delay": bli_delay_vs_raw_delay,
        "relative_reduction_bli_delay_vs_bli": bli_delay_vs_bli,
        "raw_failure_rate": float(raw_stat["failure_rate"]),
        "raw_delay_failure_rate": float(raw_delay_stat["failure_rate"]),
        "bli_failure_rate": float(bli_stat["failure_rate"]),
        "bli_delay_failure_rate": float(bli_delay_stat["failure_rate"]),
        "raw_selected_alpha": float(raw["selected_alpha"]),
        "raw_delay_selected_alpha": float(raw_delay["selected_alpha"]),
        "bli_selected_alpha": float(bli["selected_alpha"]),
        "bli_delay_selected_alpha": float(bli_delay["selected_alpha"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def print_seed_metrics(position, total, record):
    """Print every recorded modeling metric as soon as a seed finishes."""
    print(f"\n[{position:02d}/{total:02d}] seed={record['seed']:02d}", flush=True)
    for method, label in zip(("raw", "raw_delay", "bli", "bli_delay"), METHOD_LABELS):
        print(
            f"  {label}: {method}_rmse={record[f'{method}_rmse']:.6f} "
            f"{method}_failure_rate={record[f'{method}_failure_rate']:.6f} "
            f"selected_alpha={record[f'{method}_selected_alpha']:.8g}",
            flush=True,
        )
    for name, value in record.items():
        if name.startswith("relative_reduction_"):
            print(f"  {name}={100.0 * value:.6f}%", flush=True)
    print(f"  elapsed_seconds={record['elapsed_seconds']:.3f}", flush=True)


def write_seed_csv(records, path):
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_seed_comparison(records, output_path):
    import matplotlib.pyplot as plt

    seeds = np.asarray([record["seed"] for record in records], dtype=int)
    raw = np.asarray([record["raw_rmse"] for record in records], dtype=float)
    raw_delay = np.asarray([record['raw_delay_rmse'] for record in records], dtype=float)
    bli = np.asarray([record["bli_rmse"] for record in records], dtype=float)
    bli_delay = np.asarray([record['bli_delay_rmse'] for record in records], dtype=float)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.plot(seeds, raw, "o-", linewidth=1.6, label=METHOD_LABELS[0])
    axis.plot(seeds, raw_delay, "s-", linewidth=1.6, label=METHOD_LABELS[1])
    axis.plot(seeds, bli, "^-", linewidth=1.6, label=METHOD_LABELS[2])
    axis.plot(seeds, bli_delay, "d-", linewidth=1.6, label=METHOD_LABELS[3])
    axis.set_xlabel("Noise seed")
    axis.set_ylabel("Mean 20-step rolling RMSE")
    axis.set_xticks(seeds)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
