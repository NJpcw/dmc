"""End-to-end waterwheel detection, NGRC modeling, and control pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.post_detection_data import (
    build_rolling_prediction_results,
    save_matlab_plot_data,
)

from . import (
    control as control,
    detection as detection,
    modeling as modeling,
)
from .dynamics import DEFAULT_NOISE_STD, resolve_noise_std


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parent


@dataclass
class PipelineConfig:
    detector_config: detection.DetectorConfig = field(default_factory=detection.DetectorConfig)
    model_config: object = field(default_factory=modeling.make_control_ngrc_config)
    control_config: control.ControlConfig = field(default_factory=control.ControlConfig)
    modeling_duration: float = 3.0
    modeling_stride: int = 10
    modeling_noise_std: np.ndarray = field(default_factory=lambda: DEFAULT_NOISE_STD.copy())
    modeling_seed: int = 2027
    output_dir: Path = REPOSITORY_ROOT / "outputs/waterwheel/pipeline"

    def validate(self) -> None:
        self.detector_config.validate()
        self.model_config.validate()
        self.control_config.validate(self.model_config.identification_dt)
        detector_noise = resolve_noise_std(self.detector_config.noise_std)
        modeling_noise = resolve_noise_std(self.modeling_noise_std)
        control_noise = resolve_noise_std(self.control_config.measurement_noise_std)
        if not np.allclose(detector_noise, modeling_noise, rtol=0.0, atol=1e-12):
            raise ValueError("detector and modeling noise must use the same state scaling")
        if not np.allclose(detector_noise, control_noise, rtol=0.0, atol=1e-12):
            raise ValueError("detector and control noise must use the same state scaling")
        if not np.isclose(self.detector_config.dt, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("detector internal interval must be 0.001 s")
        if not np.isclose(self.model_config.dt_original, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("model internal interval must be 0.001 s")
        if not np.isclose(self.control_config.internal_dt, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("control internal interval must be 0.001 s")
        if self.modeling_stride != self.model_config.stride:
            raise ValueError("modeling stride must equal the NGRC stride")
        if self.control_config.control_stride != self.model_config.stride:
            raise ValueError("control stride must equal the NGRC stride")
        if not np.isclose(
            self.modeling_duration,
            self.model_config.training_duration,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("modeling duration must equal NGRC training duration")
        if self.model_config.poly_order != 2:
            raise ValueError("the waterwheel migration requires quadratic NGRC features")
        if (
            self.detector_config.t_end - self.detector_config.switch_time
            <= self.modeling_duration
        ):
            raise ValueError("detector record must extend beyond the post-switch modeling window")


def make_quick_config(output_dir: Path | None = None) -> PipelineConfig:
    """Return a reduced-cost configuration for integration testing."""
    root = Path(output_dir or REPOSITORY_ROOT / "outputs/waterwheel/quick")
    detector_config = replace(
        detection.DetectorConfig(),
        output_dir=root / "detection",
        t_end=9.0,
        switch_time=5.0,
        training_trajectories=4,
        validation_trajectories=3,
        trajectory_duration=5.0,
        input_window=8,
        glr_window=12,
        epochs=10,
        batch_size=128,
        conv_channels=8,
        hidden_size=12,
        lstm_layers=1,
    )
    model_config = modeling.make_ngrc_config(
        t_end=4.0,
        training_duration=1.0,
        evaluation_duration=0.5,
        external_horizon=5,
        validation_horizon=5,
        validation_anchors_per_fold=2,
        alpha_grid=np.array([1e-3, 1e-1, 1.0]),
        bli_horizons=(1, 2, 5),
        bli_alpha_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
        bli_alternating_iterations=1,
        bli_latent_max_nfev=2,
    )
    control_config = control.ControlConfig(
        control_duration=0.2,
        horizon=20,
        move_horizon=10,
        mpc_maxiter=60,
    )
    return PipelineConfig(
        detector_config=detector_config,
        model_config=model_config,
        control_config=control_config,
        modeling_duration=1.0,
        modeling_stride=10,
        modeling_noise_std=DEFAULT_NOISE_STD.copy(),
        output_dir=root,
    )


def _shared_control_noise(config: control.ControlConfig) -> np.ndarray:
    step_count = int(round(config.control_duration / config.control_dt))
    return np.random.default_rng(config.random_seed).normal(
        loc=0.0,
        scale=resolve_noise_std(config.measurement_noise_std),
        size=(step_count, 3),
    )


def _detection_times(detection_run: dict[str, object]) -> dict[str, float]:
    trajectory = detection_run["trajectory"]
    result = detection_run["result"]
    if result.alarm_index is None or result.estimated_switch_index is None:
        statistic = np.asarray(result.statistic, dtype=float)
        finite_max = float(np.nanmax(statistic)) if np.any(np.isfinite(statistic)) else np.nan
        raise RuntimeError(
            "No formal GLR alarm occurred "
            f"(max statistic={finite_max:.6g}, threshold={detection_run['calibration'].threshold:.6g})"
        )
    alarm_time = float(trajectory.sample_time[int(result.alarm_index)])
    estimated_time = float(trajectory.sample_time[int(result.estimated_switch_index)])
    true_time = float(detection_run["config"].switch_time)
    return {
        "true_switch_time": true_time,
        "estimated_switch_time": estimated_time,
        "alarm_time": alarm_time,
        "alarm_delay": alarm_time - true_time,
        "localization_error": abs(estimated_time - true_time),
        "false_alarm": alarm_time < true_time,
    }


def _control_metrics(run: dict[str, object], config: control.ControlConfig) -> dict[str, object]:
    time = np.asarray(run["t"], dtype=float)
    solve_time = np.asarray(run.get("optimizer_solve_time", []), dtype=float)
    iterations = np.asarray(run.get("optimizer_iterations", []), dtype=float)
    evaluations = np.asarray(run.get("optimizer_nfev", []), dtype=float)
    success = np.asarray(run.get('optimizer_success', run.get('solver_success', [])), dtype=bool)
    solver_failure_count = int(np.count_nonzero(~success))
    window_duration = min(2.0, float(time[-1] - time[0]))
    metrics = control.control_metric_panel(run, window_duration=window_duration)
    metrics.update(
        {
        "solver_success_fraction": float(np.mean(success)) if len(success) else np.nan,
        "solver_failure_count": solver_failure_count,
        "fallback_count": solver_failure_count,
        "optimizer_name": str(run.get("optimizer_name", "IPOPT")),
        "mean_solve_time_seconds": float(np.mean(solve_time)) if len(solve_time) else np.nan,
        "total_solve_time_seconds": float(np.sum(solve_time)) if len(solve_time) else np.nan,
        "max_solve_time_seconds": float(np.max(solve_time)) if len(solve_time) else np.nan,
        "mean_optimizer_iterations": float(np.mean(iterations)) if len(iterations) else np.nan,
        "mean_optimizer_evaluations": float(np.mean(evaluations)) if len(evaluations) else np.nan,
        "observer_update_count": int(run.get("observer_update_count", 0)),
        "observer_magnitude_saturations": int(run.get('observer_magnitude_saturations', 0)),
        "observer_slew_saturations": int(run.get("observer_slew_saturations", 0)),
        **control.constraint_violations(run, config),
        **control.steady_state_metrics(run, window_duration=window_duration),
        }
    )
    return metrics


def _external_prediction_summary(
    rolling_predictions: dict[str, object],
) -> dict[str, dict[str, float]]:
    fields = ("finite_mean", "finite_std", "finite_max", "failure_rate")
    return {
        method: {
            field: float(rolling_predictions[method]["external_stat"][field])
            for field in fields
        }
        for method in ("raw", "raw_delay", "bli")
        if method in rolling_predictions
    }


def _summary(results: dict[str, object]) -> dict[str, object]:
    models = results["models"]
    controls = results["controls"]
    config = results["config"]
    summary = {
        "system": "Malkus-Lorenz waterwheel",
        "state_order": ["omega", "a", "b"],
        "input_order": ["u"],
        "input_matrix": control.INPUT_MATRIX.tolist(),
        "q_before": 18.0,
        "q_after": 15.0,
        "noise_std": resolve_noise_std(config.modeling_noise_std).tolist(),
        "detection": results["detection_times"],
        "modeling": {
            "start_time": float(results["modeling_segment"].sample_time[0]),
            "end_time": float(results["modeling_segment"].sample_time[-1]),
            "samples": int(len(results["modeling_segment"].sample_time)),
            "poly_order": int(config.model_config.poly_order),
            "raw_alpha": float(models["raw"]["selected_alpha"]),
            "raw_delay_alpha": (
                float(models['raw_delay']['selected_alpha']) if 'raw_delay' in models else None
            ),
            "bli_alpha": float(models["bli"]["selected_alpha"]),
            "external_prediction": _external_prediction_summary(results['rolling_predictions']),
        },
        "control": {
            name: _control_metrics(run, config.control_config)
            for name, run in controls.items()
        },
        "reference": np.asarray(config.control_config.reference).tolist(),
    }
    return summary


def plot_detection(results: dict[str, object], output_path: Path) -> Path:
    run = results["detection_run"]
    times = results["detection_times"]
    trajectory = run["trajectory"]
    prediction = np.asarray(run["predictions"])
    result = run["result"]
    labels = (r"$\omega$", r"$a$", r"$b$")
    colors = ("#2563A6", "#D97706", "#0F766E")
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    for index, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(
            trajectory.sample_time,
            trajectory.sample_clean[:, index],
            color=color,
            linewidth=1.2,
            label=f"{label} clean",
        )
        axes[0].plot(
            trajectory.sample_time,
            prediction[:, index],
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.8,
        )
    axes[0].set_ylabel("State")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].plot(
        trajectory.sample_time,
        result.statistic,
        color="#374151",
        linewidth=1.2,
        label="GLR statistic",
    )
    axes[1].axhline(
        run["calibration"].threshold,
        color="#7C3AED",
        linestyle="--",
        linewidth=1.0,
        label="Threshold",
    )
    for key, color, style in (
        ("true_switch_time", "#B91C1C", "--"),
        ("estimated_switch_time", "#059669", ":"),
        ("alarm_time", "#111827", "-."),
    ):
        for axis in axes:
            axis.axvline(times[key], color=color, linestyle=style, linewidth=1.0)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("GLR")
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_modeling(results: dict[str, object], output_path: Path) -> Path:
    segment = results["modeling_segment"]
    bli_input = np.asarray(results["models"]["bli"]["X_input"])
    labels = (r"$\omega$", r"$a$", r"$b$")
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 6.6), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(segment.sample_time, segment.sample_clean[:, index], color="#111827", label="Clean")
        axis.scatter(segment.sample_time, segment.sample_noisy[:, index], s=7, color="#9CA3AF", alpha=0.65, label="Observed")
        axis.plot(segment.sample_time[: len(bli_input)], bli_input[:, index], color="#D97706", linewidth=1.1, label="BLI latent")
        axis.set_ylabel(labels[index])
        axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_control(results: dict[str, object], output_path: Path) -> Path:
    labels = (r"$\omega$", r"$a$", r"$b$")
    colors = {
        "raw_mpc": "#6B7280",
        "raw_delay_mpc": "#7C3AED",
        "bli_mpc": "#2563A6",
        "bli_adr_mpc": "#D97706",
        "raw_adr_mpc": "#059669",
    }
    display = {
        "raw_mpc": "Raw NGRC+MPC",
        "raw_delay_mpc": "Raw-delay NGRC+MPC",
        "bli_mpc": "BLI-NGRC+MPC",
        "bli_adr_mpc": "BLI-NGRC+ADR-MPC",
        "raw_adr_mpc": "Raw-NGRC+ESO-ADR+MPC",
    }
    figure = plt.figure(figsize=(11.2, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.65, 0.85), hspace=0.16)
    state_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    input_axis = figure.add_subplot(grid[1, :])
    reference = np.asarray(results["config"].control_config.reference)
    representative_time = np.asarray(next(iter(results["controls"].values()))["t"])
    steady_start = representative_time[-1] - min(
        2.0, float(representative_time[-1] - representative_time[0])
    )
    for state_index in range(3):
        for name, run in results["controls"].items():
            state_axes[state_index].plot(run["t"], run["X_true"][:, state_index], color=colors[name], linewidth=1.2, label=display[name])
        state_axes[state_index].axhline(reference[state_index], color="#B91C1C", linestyle="--", linewidth=1.0, label="Reference")
        state_axes[state_index].axvspan(
            steady_start,
            representative_time[-1],
            color="#FEF3C7",
            alpha=0.35,
            linewidth=0.0,
            label="Steady-state window" if state_index == 0 else None,
        )
        inset = state_axes[state_index].inset_axes([0.51, 0.08, 0.46, 0.38])
        for name, run in results["controls"].items():
            time = np.asarray(run["t"])
            mask = time >= steady_start - 1e-12
            error = np.asarray(run["X_true"])[mask, state_index] - reference[state_index]
            inset.plot(time[mask], error, color=colors[name], linewidth=0.8)
        inset.axhline(0.0, color="#B91C1C", linestyle="--", linewidth=0.6)
        inset.set_title("Final-window error", fontsize=6)
        inset.tick_params(labelsize=5, length=2)
        inset.grid(color="#D1D5DB", linewidth=0.35, alpha=0.5)
        state_axes[state_index].set_title(labels[state_index])
        state_axes[state_index].set_xlabel("Time")
        state_axes[state_index].grid(color="#D1D5DB", linewidth=0.5, alpha=0.6)
        state_axes[state_index].spines[["top", "right"]].set_visible(False)
    for name, run in results["controls"].items():
        input_axis.step(run["t"][:-1], run["U"], where="post", color=colors[name], linewidth=1.1, label=display[name])
    input_axis.axvspan(
        steady_start,
        representative_time[-1],
        color="#FEF3C7",
        alpha=0.35,
        linewidth=0.0,
    )
    input_axis.axhline(0.0, color="#9CA3AF", linewidth=0.7)
    input_axis.set_xlabel("Time")
    input_axis.set_ylabel(r"Single input $u$")
    input_axis.grid(color="#D1D5DB", linewidth=0.5, alpha=0.6)
    input_axis.spines[["top", "right"]].set_visible(False)
    state_axes[0].set_ylabel("State")
    state_axes[0].legend(frameon=False, fontsize=7)
    input_axis.legend(frameon=False, ncol=5, fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_outputs(results: dict[str, object]) -> dict[str, Path]:
    output_dir = Path(results["config"].output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matlab_path = save_matlab_plot_data(results, output_dir)
    rolling_path = output_dir / "rolling_prediction.png"
    rolling_figure = modeling.NGRC.plot_rollout_examples(
        results["rolling_predictions"],
        num_anchors=6,
        output_path=rolling_path,
        show=False,
    )
    plt.close(rolling_figure)
    paths = {
        "detection_figure": plot_detection(results, output_dir / "detection.png"),
        "modeling_figure": plot_modeling(results, output_dir / "modeling.png"),
        "control_figure": plot_control(results, output_dir / "control.png"),
        "rolling_prediction_figure": rolling_path,
    }
    paths["matlab_data"] = matlab_path
    return paths


def _assemble_results(
    config: PipelineConfig,
    detection_run: dict[str, object],
    detection_times: dict[str, float],
    segment,
    models: dict[str, dict[str, object]],
    rolling_predictions: dict[str, object],
    controls: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Expose the same top-level aliases as the predator integrated pipeline."""
    return {
        "config": config,
        "predictor": detection_run["predictor"],
        "detection": detection_run,
        "detection_run": detection_run,
        "detection_times": detection_times,
        "modeling": segment,
        "modeling_segment": segment,
        "model": models["bli"],
        "models": models,
        "rolling_predictions": rolling_predictions,
        "raw_mpc_control": controls["raw_mpc"],
        "raw_delay_mpc_control": controls["raw_delay_mpc"],
        "bli_mpc_control": controls["bli_mpc"],
        "bli_adr_control": controls["bli_adr_mpc"],
        "raw_adr_control": controls["raw_adr_mpc"],
        "ordinary_control": controls["bli_mpc"],
        "eso_control": controls["bli_adr_mpc"],
        "controls": controls,
    }


def run_control_benchmarks_from_modeling_handoff(
    models: dict[str, dict[str, object]],
    modeling_segment,
    control_config: control.ControlConfig,
) -> dict[str, dict[str, object]]:
    """Run every controller with one shared realized measurement-noise record."""
    shared_noise = _shared_control_noise(control_config)
    return {
        "raw_mpc": control.simulate_controller_from_handoff(
            models["raw"], modeling_segment, control_config, "mpc", shared_noise
        ),
        "raw_delay_mpc": control.simulate_controller_from_handoff(
            models["raw_delay"],
            modeling_segment,
            control_config,
            "mpc",
            shared_noise,
        ),
        "bli_mpc": control.simulate_controller_from_handoff(
            models["bli"], modeling_segment, control_config, "mpc", shared_noise
        ),
        "bli_adr_mpc": control.simulate_controller_from_handoff(
            models["bli"], modeling_segment, control_config, "adr_mpc", shared_noise
        ),
        "raw_adr_mpc": control.simulate_controller_from_handoff(
            models["raw"], modeling_segment, control_config, "adr_mpc", shared_noise
        ),
    }


def run_pipeline(
    config: PipelineConfig | None = None,
    device_name: str = "cpu",
    save_results: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    if config is None:
        config = PipelineConfig()
    config.validate()
    detection_run = detection.run_detection(config.detector_config, device_name=device_name)
    detection_times = _detection_times(detection_run)
    segment = modeling.prepare_modeling_segment(
        detection_run["trajectory"],
        detection_run["result"],
        duration=config.modeling_duration,
        modeling_stride=config.modeling_stride,
        noise_std=config.modeling_noise_std,
        seed=config.modeling_seed,
    )
    models = modeling.fit_models(segment, config.model_config)
    rolling_predictions = build_rolling_prediction_results(
        trajectory=detection_run["trajectory"],
        detection_result=detection_run["result"],
        models=models,
        config=config,
    )
    rolling_predictions["state_names"] = ("omega", "a", "b")
    controls = run_control_benchmarks_from_modeling_handoff(models, segment, config.control_config)
    results = _assemble_results(
        config,
        detection_run,
        detection_times,
        segment,
        models,
        rolling_predictions,
        controls,
    )
    results["summary"] = _summary(results)
    results["paths"] = save_outputs(results) if save_results else {}
    if verbose:
        print(json.dumps(results["summary"], indent=2, ensure_ascii=False))
        if save_results:
            print(f"MATLAB plot data: {results['paths']['matlab_data']}")
    return results
