"""Post-alarm data handoff and three-state Raw/BLI-NGRC fitting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from common.modeling_control import ngrc_modeling as NGRC

from .dynamics import CORE, DEFAULT_INITIAL_STATE, DEFAULT_NOISE_STD


PostAlarmSegment = CORE.PostAlarmSegment


def make_ngrc_config(**overrides):
    """Return the quadratic three-state NGRC configuration for the waterwheel."""
    config = NGRC.ExperimentConfig(
        x0=DEFAULT_INITIAL_STATE.copy(),
        t_end=12.0,
        dt_original=0.001,
        stride=10,
        noise_std=DEFAULT_NOISE_STD.copy(),
        random_seed=40_001,
        training_duration=3.0,
        evaluation_duration=5.0,
        k=2,
        s=1,
        poly_order=2,
        feature_clip=50.0,
        max_abs_state=100.0,
        bli_alpha_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    )
    if overrides:
        config = replace(config, **overrides)
    config.validate()
    return config


def make_control_ngrc_config():
    """Return the validation-selected BLI configuration used elsewhere."""
    return make_ngrc_config()


def prepare_modeling_segment(
    trajectory,
    detection_result,
    duration: float = 3.0,
    modeling_stride: int = 10,
    noise_std: np.ndarray | float = DEFAULT_NOISE_STD,
    seed: int = 40_001,
) -> PostAlarmSegment:
    """Sample a post-alarm window from the detector's exact internal record."""
    if detection_result.alarm_index is None:
        raise ValueError("a formal detection alarm is required before modeling")
    alarm_sample_index = int(detection_result.alarm_index)
    if alarm_sample_index < 0 or alarm_sample_index >= len(trajectory.sample_time):
        raise ValueError("alarm index lies outside the sampled trajectory")
    alarm_internal_index = int(trajectory.sample_indices[alarm_sample_index])
    internal_time = np.asarray(trajectory.internal_time, dtype=float)
    internal_clean = np.asarray(trajectory.internal_clean, dtype=float)
    if len(internal_time) < 2 or internal_clean.shape != (len(internal_time), 3):
        raise ValueError("trajectory must contain a three-state internal record")
    internal_dt = float(internal_time[1] - internal_time[0])
    return CORE.build_post_alarm_segment(
        internal_time=internal_time,
        internal_clean=internal_clean,
        alarm_internal_index=alarm_internal_index,
        alarm_observation=np.asarray(trajectory.sample_noisy[alarm_sample_index], dtype=float),
        duration=duration,
        internal_dt=internal_dt,
        modeling_stride=modeling_stride,
        noise_std=noise_std,
        seed=seed,
    )


def fit_models(
    segment: PostAlarmSegment,
    config=None,
    methods: tuple[str, ...] = ("raw", "raw_delay", "bli"),
) -> dict[str, dict[str, object]]:
    """Fit the requested NGRC models to the shared post-alarm record."""
    if config is None:
        config = make_ngrc_config()
    config.validate()
    if not methods:
        raise ValueError("at least one modeling method is required")
    models = {}
    for method in methods:
        models[method] = NGRC.train_method_from_observations(
            segment.sample_time,
            segment.sample_noisy,
            config,
            method=method,
        )
    return models


def predict_one_step(model: dict[str, object], history: np.ndarray) -> np.ndarray:
    return np.asarray(NGRC.predict_one_step_original(model, history), dtype=float)


def minimum_history_length(model: dict[str, object]) -> int:
    return (int(model["k"]) - 1) * int(model["s"]) + 1
