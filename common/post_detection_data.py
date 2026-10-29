"""Shared detector-to-model handoff and rolling evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from .detection.residual_glr_detection import build_post_alarm_segment
from .modeling_control import ngrc_modeling as modeling


def prepare_modeling_segment(
    trajectory,
    detection_result,
    duration=4.0,
    modeling_stride=10,
    noise_std=0.05,
    seed=40_001,
):
    """Build a causal post-alarm modeling record from a detected trajectory."""
    if detection_result.alarm_index is None:
        raise ValueError("A formal detection alarm is required before modeling.")
    alarm_sample_index = int(detection_result.alarm_index)
    if alarm_sample_index < 0 or alarm_sample_index >= len(trajectory.sample_time):
        raise ValueError("Detection alarm index lies outside the sampled trajectory.")
    alarm_internal_index = int(trajectory.sample_indices[alarm_sample_index])
    if len(trajectory.internal_time) < 2:
        raise ValueError("Internal trajectory is too short.")
    internal_dt = float(trajectory.internal_time[1] - trajectory.internal_time[0])
    return build_post_alarm_segment(
        internal_time=trajectory.internal_time,
        internal_clean=trajectory.internal_clean,
        alarm_internal_index=alarm_internal_index,
        alarm_observation=trajectory.sample_noisy[alarm_sample_index],
        duration=duration,
        internal_dt=internal_dt,
        modeling_stride=modeling_stride,
        noise_std=noise_std,
        seed=seed,
    )


def build_rolling_prediction_results(trajectory, detection_result, models, config):
    """Evaluate requested Raw-family and BLI models at shared roll origins."""
    methods = tuple(models)
    missing = [method for method in ("raw", "bli") if method not in models]
    if missing:
        raise ValueError(f"Missing fitted models for rolling prediction: {missing}")
    unsupported = [
        method
        for method in methods
        if method not in ("raw", "raw_delay", "bli", "bli_delay")
    ]
    if unsupported:
        raise ValueError(f"Unsupported rolling prediction models: {unsupported}")

    model_config = config.model_config
    rollout_duration = (
        config.modeling_duration
        + model_config.evaluation_duration
        + model_config.external_horizon * model_config.identification_dt
    )
    extended_segment = prepare_modeling_segment(
        trajectory=trajectory,
        detection_result=detection_result,
        duration=rollout_duration,
        modeling_stride=config.modeling_stride,
        noise_std=config.modeling_noise_std,
        seed=config.modeling_seed,
    )
    training_points = model_config.training_samples + 1
    extended_observations = np.asarray(extended_segment.sample_noisy, dtype=float)

    rolling_results = {
        "config": model_config,
        "t": extended_segment.sample_time,
        "X_clean": extended_segment.sample_clean,
        "X_noisy": extended_observations,
        "methods": methods,
        "evaluation_start_time": float(extended_segment.sample_time[model_config.training_samples]),
        "observation_segment": extended_segment,
    }
    for method in methods:
        fitted_input = np.asarray(models[method]["X_input"], dtype=float)
        if len(fitted_input) != training_points:
            raise ValueError(f'{method} model input must contain the complete training window.')
        fitted_observations = (
            np.asarray(models[method]["X_observed"], dtype=float)
            if method in ("bli", "bli_delay")
            else fitted_input
        )
        if not np.allclose(
            fitted_observations,
            extended_observations[:training_points],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{method} rolling observations do not continue the fitted trajectory."
            )
        rolling_model = dict(models[method])
        if method in ("bli", "bli_delay"):
            rolling_model["X_input"] = np.vstack(
                [fitted_input, extended_observations[training_points:]]
            )
            rolling_model["X_observed"] = extended_observations.copy()
            rolling_model["fit_sample_count"] = int(len(extended_observations))
        else:
            rolling_model["X_input"] = extended_observations.copy()
        rolling_results[method] = rolling_model
    rolling_results["X_bli"] = rolling_results["bli"]["X_input"]

    return modeling.evaluate_external_window(rolling_results, config=model_config, methods=methods)



def save_matlab_plot_data(results, output_dir):
    """Export one integrated run; rows are samples and indices are MATLAB-based."""
    run = results["detection"]
    trajectory, detected = run["trajectory"], run["result"]
    config = results["config"]
    segment = results["modeling"]
    rolling = results["rolling_predictions"]
    state_names = rolling.get("state_names", ("x", "y"))
    times = results.get("detection_times", run)
    detection_data = {
        "time": trajectory.sample_time,
        "clean": trajectory.sample_clean,
        "noisy": trajectory.sample_noisy,
        "prediction": run["predictions"],
        "innovations": run["innovations"],
        "innovation_valid": np.isfinite(run["innovations"]).all(axis=1),
        "energy": detected.energy,
        "glr": detected.statistic,
        "glr_valid": np.isfinite(detected.statistic),
        "threshold": run["calibration"].threshold,
        "alarm": detected.alarm,
        "alarm_index": int(detected.alarm_index) + 1,
        "estimated_switch_index": int(detected.estimated_switch_index) + 1,
    }
    for key in ("true_switch_time", "estimated_switch_time", "alarm_time"):
        detection_data[key] = times[key]

    # Use the same six origins and deterministic rollout routine as the figure.
    all_anchors = np.asarray(rolling["external_anchors"], dtype=int)
    positions = np.unique(np.linspace(0, len(all_anchors) - 1, 6, dtype=int))
    anchors = all_anchors[positions]
    horizon = rolling["config"].external_horizon
    windows = [slice(int(anchor), int(anchor) + horizon + 1) for anchor in anchors]
    modeling_data = {
        "time": segment.sample_time,
        "clean": segment.sample_clean,
        "noisy": segment.sample_noisy,
        "rolling_time": rolling["t"],
        "rolling_clean": rolling["X_clean"],
        "rolling_noisy": rolling["X_noisy"],
        "evaluation_start_time": rolling["evaluation_start_time"],
        "anchor_indices": anchors + 1,
        "external_anchor_indices": all_anchors + 1,
        "prediction_time": np.stack([rolling["t"][window] for window in windows], axis=1),
        "prediction_clean": np.stack(
            [rolling["X_clean"][window] for window in windows], axis=2
        ),
        "horizon": horizon,
        "methods": np.asarray(rolling["methods"], dtype=object),
    }
    for method in rolling["methods"]:
        predictions = [
            modeling.rollout_from_anchor(
                rolling[method], rolling["X_clean"], int(anchor), horizon,
                max_abs_state=rolling["config"].max_abs_state,
            )[1]
            for anchor in anchors
        ]
        fitted_input = results["models"][method]["X_input"]
        modeling_data[method] = {
            "input_time": segment.sample_time[:len(fitted_input)],
            "input": fitted_input,
            "prediction": np.stack(predictions, axis=2),
            "external_rmse": rolling[method]["external_rmse_values"],
        }

    controls = {}
    for name in ("raw_mpc", "raw_delay_mpc", "bli_mpc", "bli_adr", "raw_adr"):
        control_run = results[f"{name}_control"]
        controls[name] = dict(control_run, t_input=control_run["t"][:-1])
    control_config = config.control_config
    control_settings = {
        key: getattr(control_config, key)
        for key in ("reference", "u_min", "u_max", "du_max", "steady_state_window")
    }
    path = Path(output_dir) / "plot_data.mat"
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {
        "state_names": np.asarray(state_names, dtype=object),
        "detection": detection_data,
        "modeling": modeling_data,
        "control": controls,
        "control_settings": control_settings,
    }, do_compression=True, long_field_names=True, oned_as="column")
    return path
