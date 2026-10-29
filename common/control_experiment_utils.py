"""Shared helpers for terminal-only, paired multi-seed control trials.

The experiment entry points live in ``predator_prey/run_control_trials.py``
and ``waterwheel/run_control_trials.py``.  This module does not write files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .modeling_control import disturbance_rejection_mpc as CONTROL


APPROVED_METRICS = (
    "control_total_variation",
    "final_window_rmse",
    "iae",
    "itae",
    "control_energy",
    "solver_success_fraction",
)

METRIC_LABELS = {
    "control_total_variation": "TV",
    "final_window_rmse": "final_2s_rmse",
    "iae": "IAE",
    "itae": "ITAE",
    "control_energy": "energy",
    "solver_success_fraction": "solver_success",
}


def _positive_sample_count(sample_count: int) -> int:
    if not isinstance(sample_count, (int, np.integer)) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    return int(sample_count)


def _standard_deviation_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if (
        vector.ndim != 1
        or len(vector) == 0
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
    ):
        raise ValueError(f"{name} must be a finite nonnegative scalar or vector")
    return vector


def spawn_trial_generators(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Create independent measurement and process generators for one trial."""
    if not isinstance(seed, (int, np.integer)) or int(seed) <= 0:
        raise ValueError("seed must be a positive integer")
    measurement_sequence, process_sequence = np.random.SeedSequence(int(seed)).spawn(2)
    return (np.random.default_rng(measurement_sequence), np.random.default_rng(process_sequence))


def generate_measurement_noise(
    rng: np.random.Generator,
    sample_count: int,
    noise_std,
) -> np.ndarray:
    """Generate one measurement-noise row per feedback sample."""
    count = _positive_sample_count(sample_count)
    standard_deviation = _standard_deviation_vector(noise_std, "noise_std")
    return rng.normal(loc=0.0, scale=standard_deviation, size=(count, len(standard_deviation)))


def generate_ou_disturbance(
    rng: np.random.Generator,
    sample_count: int,
    dt: float,
    tau: float,
    stationary_std,
) -> np.ndarray:
    """Generate a stationary Ornstein-Uhlenbeck Gaussian disturbance path."""
    count = _positive_sample_count(sample_count)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    standard_deviation = _standard_deviation_vector(stationary_std, 'stationary_std')
    path = np.empty((count, len(standard_deviation)), dtype=float)
    path[0] = rng.normal(loc=0.0, scale=standard_deviation)
    rho = float(np.exp(-float(dt) / float(tau)))
    innovation_std = standard_deviation * np.sqrt(1.0 - rho**2)
    for index in range(1, count):
        path[index] = rho * path[index - 1] + rng.normal(loc=0.0, scale=innovation_std)
    return path


def extract_control_metrics(
    simulation: Mapping[str, object],
    window_duration: float = 2.0,
) -> dict[str, float]:
    """Extract only the metrics approved for the repeated-control experiment."""
    panel = CONTROL.control_metric_panel(simulation, window_duration=float(window_duration))
    optimizer_success = np.asarray(simulation.get('optimizer_success', []), dtype=bool)
    if optimizer_success.ndim != 1 or len(optimizer_success) == 0:
        raise ValueError("optimizer_success must be a nonempty one-dimensional array")
    metrics = {
        "control_total_variation": float(panel["control_total_variation"]),
        "final_window_rmse": float(panel["final_window_rmse"]),
        "iae": float(panel["iae"]),
        "itae": float(panel["itae"]),
        "control_energy": float(panel["control_energy"]),
        "solver_success_fraction": float(np.mean(optimizer_success)),
    }
    if tuple(metrics) != APPROVED_METRICS or not np.all(np.isfinite(list(metrics.values()))):
        raise ValueError("control metrics must contain finite approved values")
    return metrics


def summarize_metric_values(values: Iterable[float]) -> dict[str, object]:
    """Summarize finite trials with a two-sided Student-t confidence interval."""
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("metric values must be one-dimensional")
    finite = array[np.isfinite(array)]
    count = int(len(finite))
    if count == 0:
        return {
            "valid_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "mean_95_percent_ci": [float("nan"), float("nan")],
        }
    mean = float(np.mean(finite))
    if count == 1:
        standard_deviation = 0.0
        lower = upper = mean
    else:
        standard_deviation = float(np.std(finite, ddof=1))
        critical_value = float(student_t.ppf(0.975, df=count - 1))
        half_width = critical_value * standard_deviation / np.sqrt(count)
        lower = mean - half_width
        upper = mean + half_width
    return {
        "valid_count": count,
        "mean": mean,
        "std": standard_deviation,
        "mean_95_percent_ci": [float(lower), float(upper)],
    }


def _method_metrics_or_nan(
    method_metrics: Mapping[str, Mapping[str, float]],
    method: str,
) -> Mapping[str, float]:
    return method_metrics.get(method, {metric: float('nan') for metric in APPROVED_METRICS})


def print_trial_metrics(
    position: int,
    total: int,
    seed: int,
    method_metrics: Mapping[str, Mapping[str, float]],
    method_order: Sequence[str],
) -> None:
    """Print all approved metrics for one paired seed."""
    print(f"\n[{position:02d}/{total:02d}] seed={seed:02d}", flush=True)
    for method in method_order:
        metrics = _method_metrics_or_nan(method_metrics, method)
        values = " ".join(
            f"{METRIC_LABELS[metric]}={float(metrics[metric]):.6f}"
            for metric in APPROVED_METRICS
        )
        print(f"  {method:<28s} {values}", flush=True)


def print_aggregate_metrics(
    records: Sequence[Mapping[str, Mapping[str, float]]],
    method_order: Sequence[str],
) -> None:
    """Print finite-only aggregate statistics for every method and metric."""
    print("\nAggregate statistics", flush=True)
    for method in method_order:
        print(f"\n{method}", flush=True)
        for metric in APPROVED_METRICS:
            summary = summarize_metric_values(
                _method_metrics_or_nan(record, method)[metric]
                for record in records
            )
            interval = summary["mean_95_percent_ci"]
            print(
                f"  {METRIC_LABELS[metric]:<19s} "
                f"n={summary['valid_count']:2d} "
                f"mean={summary['mean']:.6f} "
                f"std={summary['std']:.6f} "
                f"95% CI=[{interval[0]:.6f}, {interval[1]:.6f}]",
                flush=True,
            )


def _process_index(time: float, handoff_time: float, internal_dt: float, path_length: int) -> int:
    index = int(round((float(time) - float(handoff_time)) / float(internal_dt)))
    if index < 0 or index >= path_length:
        raise IndexError("Process-disturbance path does not cover the requested time")
    return index


def _nan_metrics() -> dict[str, float]:
    return {metric: float("nan") for metric in APPROVED_METRICS}
