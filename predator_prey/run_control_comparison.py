"""Run the five-method Raw/Raw-delay/BLI NGRC control comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from predator_prey.pipeline import (
    CONTROL,
    IntegratedPipelineConfig,
    run_integrated_pipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONTROL_METHODS = (
    ("Raw-NGRC+MPC", "raw_mpc_control"),
    ("Raw-delay-NGRC+MPC", "raw_delay_mpc_control"),
    ("BLI-NGRC+MPC", "bli_mpc_control"),
    ("BLI-NGRC+ESO-ADR+MPC", "bli_adr_control"),
    ("Raw-NGRC+ESO-ADR+MPC", "raw_adr_control"),
)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def summarize_control_method(simulation, control_config):
    controls = np.asarray(simulation["U_actual"], dtype=float)
    optimizer_success = np.asarray(simulation["optimizer_success"], dtype=bool)
    optimizer_solve_time = np.asarray(simulation['optimizer_solve_time'], dtype=float)
    solver_failure_count = int(np.count_nonzero(~optimizer_success))
    previous_zero = np.zeros((1, controls.shape[1]), dtype=float)
    increments = np.diff(np.vstack([previous_zero, controls]), axis=0)
    u_min = np.asarray(control_config.u_min, dtype=float)
    u_max = np.asarray(control_config.u_max, dtype=float)
    du_max = np.asarray(control_config.du_max, dtype=float)
    input_violations = (controls < u_min - 1e-9) | (controls > u_max + 1e-9)
    slew_violations = np.abs(increments) > du_max + 1e-9
    disturbance_estimate = np.asarray(simulation["D_hat"], dtype=float)
    disturbance_true = np.asarray(simulation["D_true"], dtype=float)
    aligned_estimate = disturbance_estimate[: len(disturbance_true)]
    return {
        **CONTROL.control_metric_panel(
            simulation, window_duration=control_config.steady_state_window
        ),
        "selected_alpha": _finite_or_none(simulation["selected_alpha"]),
        "tracking_rmse": _finite_or_none(CONTROL.tracking_rmse(simulation)),
        "steady_state_rmse": _finite_or_none(
            CONTROL.steady_state_rmse(
                simulation, window_duration=control_config.steady_state_window
            )
        ),
        "final_tracking_error": _finite_or_none(CONTROL.final_tracking_error(simulation)),
        "control_energy": _finite_or_none(CONTROL.control_energy(simulation)),
        "control_increment_rms": _finite_or_none(CONTROL.control_increment_rms(simulation)),
        "optimizer_success_rate": _finite_or_none(np.mean(optimizer_success)),
        "solver_failure_count": solver_failure_count,
        "fallback_count": solver_failure_count,
        "optimizer_name": str(simulation["optimizer_name"]),
        "mean_mpc_solve_time_seconds": _finite_or_none(np.mean(optimizer_solve_time)),
        "total_mpc_solve_time_seconds": _finite_or_none(np.sum(optimizer_solve_time)),
        "max_solve_time_seconds": _finite_or_none(np.max(optimizer_solve_time)),
        "mean_ipopt_iterations": _finite_or_none(np.mean(simulation['optimizer_iterations'])),
        "mean_objective_evaluations": _finite_or_none(np.mean(simulation['optimizer_nfev'])),
        "maximum_absolute_input": _finite_or_none(np.max(np.abs(controls))),
        "maximum_absolute_slew": _finite_or_none(np.max(np.abs(increments))),
        "input_violation_count": int(np.count_nonzero(input_violations)),
        "slew_violation_count": int(np.count_nonzero(slew_violations)),
        "constraint_violation_count": int(
            np.count_nonzero(input_violations) + np.count_nonzero(slew_violations)
        ),
        "observer_update_count": int(simulation["observer_update_count"]),
        "maximum_estimated_disturbance": _finite_or_none(np.max(np.abs(disturbance_estimate))),
        "disturbance_estimation_rmse": _finite_or_none(
            np.sqrt(np.mean((aligned_estimate - disturbance_true) ** 2))
        ),
        "horizon": int(simulation["horizon"]),
        "move_horizon": int(simulation["move_horizon"]),
    }


def evaluate_numerical_validity(method_summaries):
    methods = [method_summaries[label] for label, _ in CONTROL_METHODS]
    optimizer_success = all(method['optimizer_success_rate'] >= 0.95 for method in methods)
    zero_violations = all(method['constraint_violation_count'] == 0 for method in methods)
    return {
        "optimizer_success_at_least_95_percent": bool(optimizer_success),
        "zero_constraint_violations": bool(zero_violations),
        "numerically_valid": bool(optimizer_success and zero_violations),
    }


def run_comparison(output_dir=None, verbose=True, checkpoint_epoch=100, control_seed=10):
    output_dir = Path(output_dir or REPOSITORY_ROOT / 'outputs/predator_prey/control_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    config = IntegratedPipelineConfig()
    if checkpoint_epoch is not None:
        config.detector_config = replace(
            config.detector_config, checkpoint_epoch=int(checkpoint_epoch)
        )
    config.control_config.random_seed = int(control_seed)

    started = time.perf_counter()
    results = run_integrated_pipeline(
        config=config, make_plots=False, verbose=verbose, matlab_output_dir=output_dir
    )
    elapsed = time.perf_counter() - started

    figure = CONTROL.plot_control_comparison(
        results["bli_mpc_control"],
        results["bli_adr_control"],
        raw_mpc=results["raw_mpc_control"],
        raw_delay_mpc=results["raw_delay_mpc_control"],
        raw_adr_mpc=results["raw_adr_control"],
        output_path=output_dir / "control_comparison.png",
        show=False,
    )
    CONTROL.plt.close(figure)

    method_summaries = {
        label: summarize_control_method(results[result_key], config.control_config)
        for label, result_key in CONTROL_METHODS
    }
    numerical_validity = evaluate_numerical_validity(method_summaries)
    summary = {
        "conditions": {
            "measurement_noise_std": config.control_config.measurement_noise_std,
            "modeling_noise_std": config.modeling_noise_std,
            "training_duration": config.modeling_duration,
            "identification_dt": config.model_config.identification_dt,
            "modeling_rollout_horizon": config.model_config.external_horizon,
            "control_duration": config.control_config.control_duration,
            "mpc_horizon": config.control_config.horizon,
            "methods": [label for label, _ in CONTROL_METHODS],
            "shared_measurement_noise": True,
            "shared_handoff": True,
            "control_seed": config.control_config.random_seed,
            "final_benchmark_seed": 10,
            "is_final_benchmark": config.control_config.random_seed == 10,
            "checkpoint_epoch": config.detector_config.checkpoint_epoch,
        },
        "detection": {
            "true_switch_time": results["detection"]["true_switch_time"],
            "estimated_switch_time": results["detection"]["estimated_switch_time"],
            "alarm_time": results["detection"]["alarm_time"],
            "alarm_delay": results["detection"]["alarm_delay"],
            "localization_error": results["detection"]["localization_error"],
            "false_alarm": results["detection"]["false_alarm"],
        },
        "elapsed_seconds": elapsed,
        "methods": method_summaries,
        "numerical_validity": numerical_validity,
        "numerically_valid": numerical_validity["numerically_valid"],
    }
    if verbose:
        print(f"Elapsed: {elapsed:.3f} s")
        print(f"Artifacts: {output_dir}")
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/predator_prey/control_comparison",
        help="Directory for MATLAB plot data and the control comparison figure.",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=100,
        help="Load an existing detector checkpoint instead of retraining.",
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=10,
        help="Measurement-noise seed for the comparison.",
    )
    arguments = parser.parse_args()
    run_comparison(
        output_dir=arguments.output_dir,
        verbose=True,
        checkpoint_epoch=arguments.checkpoint_epoch,
        control_seed=arguments.control_seed,
    )


if __name__ == "__main__":
    main()
