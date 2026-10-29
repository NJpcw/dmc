"""Run the 20-seed four-method NGRC waterwheel ablation comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from common import prediction_metrics as shared_comparison
from common.post_detection_data import build_rolling_prediction_results

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parent

from waterwheel import (
    modeling as modeling,
    pipeline as pipeline,
    dynamics as system,
)
from waterwheel.dynamics import resolve_noise_std


DEFAULT_SEEDS = range(1, 21)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/waterwheel/model_comparison"
FIXED_START_TIME = 100.05
MODELING_NOISE_SCALE = 1.0


def build_seed_record(
    seed: int,
    rolling_results: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build one waterwheel record with the shared comparison schema."""
    return shared_comparison.build_seed_record(
        seed=seed,
        results=rolling_results,
        elapsed_seconds=elapsed_seconds,
    )


def run_seed_comparison(
    seed: int,
    clean_trajectory,
    start_sample_index: int,
    config: pipeline.PipelineConfig,
) -> dict[str, object]:
    """Fit and evaluate the four NGRC ablations on one noisy record."""
    seed_config = replace(config, modeling_seed=int(seed))
    sample_noisy = clean_trajectory.sample_clean.copy()
    first_rng = np.random.default_rng(1_000_000 + int(seed))
    sample_noisy[start_sample_index] += first_rng.normal(
        0.0,
        resolve_noise_std(seed_config.modeling_noise_std),
    )
    trajectory = replace(clean_trajectory, sample_noisy=sample_noisy)
    fixed_start_result = SimpleNamespace(alarm_index=int(start_sample_index))
    segment = modeling.prepare_modeling_segment(
        trajectory,
        fixed_start_result,
        duration=seed_config.modeling_duration,
        modeling_stride=seed_config.modeling_stride,
        noise_std=seed_config.modeling_noise_std,
        seed=seed_config.modeling_seed,
    )
    models = modeling.fit_models(
        segment,
        config=seed_config.model_config,
        methods=modeling.NGRC.DEFAULT_COMPARISON_METHODS,
    )
    return build_rolling_prediction_results(
        trajectory=trajectory,
        detection_result=fixed_start_result,
        models=models,
        config=seed_config,
    )


def _validated_seeds(seeds) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values or any(seed <= 0 for seed in values):
        raise ValueError("At least one positive seed is required.")
    return values


def run_comparison(
    seeds=DEFAULT_SEEDS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> dict[str, object]:
    """Run the fixed-start waterwheel modeling comparison."""
    seeds = _validated_seeds(seeds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = pipeline.PipelineConfig()
    config.modeling_noise_std = (
        resolve_noise_std(config.modeling_noise_std) * MODELING_NOISE_SCALE
    )
    clean_trajectory = system.simulate_sampled_trajectory(
        t_end=110.0,
        noise_std=np.zeros(3),
        seed=0,
        switch_time=100.0,
    )
    start_sample_index = int(
        np.searchsorted(clean_trajectory.sample_time, FIXED_START_TIME, side="left")
    )
    if start_sample_index >= len(clean_trajectory.sample_time) or not np.isclose(
        clean_trajectory.sample_time[start_sample_index],
        FIXED_START_TIME,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("The fixed modeling start time is off the observation grid.")

    records = []
    started = time.perf_counter()
    if verbose:
        print(f"Fixed modeling start time: {FIXED_START_TIME}")
        print(f"Training duration: {config.modeling_duration}")
        print(f"Modeling noise std: {config.modeling_noise_std.tolist()}")
        print(
            "BLI settings: "
            f"whittaker={config.model_config.bli_whittaker_penalty}, "
            f"smoothness={config.model_config.bli_latent_smoothness_weight}, "
            f"dynamics={config.model_config.bli_latent_dynamics_weight}, "
            f"alternations={config.model_config.bli_alternating_iterations}, "
            f"max_nfev={config.model_config.bli_latent_max_nfev}"
        )
    for position, seed in enumerate(seeds, start=1):
        seed_started = time.perf_counter()
        rolling_results = run_seed_comparison(seed, clean_trajectory, start_sample_index, config)
        record = build_seed_record(
            seed=seed,
            rolling_results=rolling_results,
            elapsed_seconds=time.perf_counter() - seed_started,
        )
        records.append(record)
        if verbose:
            shared_comparison.print_seed_metrics(position, len(seeds), record)

    aggregate = shared_comparison.summarize_seed_records(records)
    summary = {
        "conditions": {
            "system": "Malkus-Lorenz waterwheel",
            "fixed_start_time": FIXED_START_TIME,
            "fixed_start_state": clean_trajectory.sample_clean[start_sample_index].tolist(),
            "noise_std": resolve_noise_std(config.modeling_noise_std).tolist(),
            "noise_scale_vs_nominal": MODELING_NOISE_SCALE,
            "bli_hyperparameters": {
                "alternating_iterations": config.model_config.bli_alternating_iterations,
                "whittaker_penalty": config.model_config.bli_whittaker_penalty,
                "latent_smoothness_weight": (
                    config.model_config.bli_latent_smoothness_weight
                ),
                "latent_dynamics_weight": (
                    config.model_config.bli_latent_dynamics_weight
                ),
                "endpoint_weight": config.model_config.bli_endpoint_weight,
                "latent_max_nfev": config.model_config.bli_latent_max_nfev,
                "objective_tolerance": config.model_config.bli_objective_tolerance,
            },
            "training_duration_seconds": float(config.modeling_duration),
            "identification_dt": float(config.model_config.identification_dt),
            "rollout_horizon": int(config.model_config.external_horizon),
            "external_anchors_per_seed": int(config.model_config.external_anchors),
            "seeds": list(seeds),
            "methods": shared_comparison.METHOD_LABELS,
        },
        "aggregate": aggregate,
        "elapsed_seconds": float(time.perf_counter() - started),
        "seed_records": records,
    }
    shared_comparison.plot_seed_comparison(records, output_dir / 'seed_rmse_comparison.png')
    if verbose:
        print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
        print(f"Artifacts: {output_dir}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--seed-end", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.seed_start <= 0 or arguments.seed_end < arguments.seed_start:
        raise ValueError("Seed range must be positive and ordered.")
    run_comparison(
        seeds=range(arguments.seed_start, arguments.seed_end + 1),
        output_dir=arguments.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
