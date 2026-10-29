"""Simulation, preprocessing, predictor, and GLR detection primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EcosystemParameters:
    r: float
    K: float
    alpha: float
    beta: float
    d: float


NORMAL_PARAMETERS = EcosystemParameters(r=0.5, K=50.0, alpha=0.02, beta=0.8, d=0.15)
ABNORMAL_PARAMETERS = EcosystemParameters(r=0.3, K=50.0, alpha=0.03, beta=0.8, d=0.3)


@dataclass(frozen=True)
class SampledTrajectory:
    internal_time: np.ndarray
    internal_clean: np.ndarray
    sample_indices: np.ndarray
    sample_time: np.ndarray
    sample_clean: np.ndarray
    sample_noisy: np.ndarray


@dataclass(frozen=True)
class PostAlarmSegment:
    alarm_internal_index: int
    internal_time: np.ndarray
    internal_clean: np.ndarray
    sample_indices: np.ndarray
    sample_time: np.ndarray
    sample_clean: np.ndarray
    sample_noisy: np.ndarray


def resolve_state_noise(noise_std: float | np.ndarray, state_dim: int) -> np.ndarray:
    values = np.asarray(noise_std, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(state_dim, float(values), dtype=np.float64)
    if values.shape != (state_dim,):
        raise ValueError("noise_std must be scalar or contain one value per state channel")
    if not np.isfinite(values).all():
        raise ValueError("noise_std must be finite")
    return values


def build_post_alarm_segment(
    internal_time: np.ndarray,
    internal_clean: np.ndarray,
    alarm_internal_index: int,
    alarm_observation: np.ndarray,
    duration: float,
    internal_dt: float,
    modeling_stride: int,
    noise_std: float | np.ndarray,
    seed: int,
) -> PostAlarmSegment:
    """Create the shared detector-to-model handoff for any state dimension."""
    internal_time = np.asarray(internal_time, dtype=np.float64)
    internal_clean = np.asarray(internal_clean, dtype=np.float64)
    alarm_observation = np.asarray(alarm_observation, dtype=np.float64)
    if internal_time.ndim != 1 or internal_clean.ndim != 2:
        raise ValueError("internal time and state records must be one- and two-dimensional")
    if len(internal_time) != len(internal_clean) or len(internal_time) < 2:
        raise ValueError("internal time and state records must have equal non-trivial length")
    if alarm_observation.shape != (internal_clean.shape[1],):
        raise ValueError("alarm_observation must contain one value per state channel")
    if not isinstance(alarm_internal_index, (int, np.integer)) or alarm_internal_index < 0:
        raise ValueError("alarm_internal_index must be a non-negative integer")
    if internal_dt <= 0 or duration <= 0:
        raise ValueError("time settings must be positive")
    resolved_noise = resolve_state_noise(noise_std, internal_clean.shape[1])
    if np.any(resolved_noise < 0):
        raise ValueError("noise_std cannot be negative")
    if not isinstance(modeling_stride, (int, np.integer)) or modeling_stride < 1:
        raise ValueError("modeling_stride must be a positive integer")
    if not np.allclose(np.diff(internal_time), internal_dt, rtol=0.0, atol=1e-12):
        raise ValueError("internal_time must be uniformly sampled at internal_dt")

    internal_steps = int(round(duration / internal_dt))
    if not np.isclose(internal_steps * internal_dt, duration, rtol=0.0, atol=1e-12):
        raise ValueError("duration must be an integer multiple of internal_dt")
    if internal_steps % modeling_stride:
        raise ValueError("duration must contain an integer number of modeling intervals")
    end_index = alarm_internal_index + internal_steps
    if alarm_internal_index >= len(internal_time) or end_index >= len(internal_time):
        raise ValueError("the internal record is too short for post-alarm modeling")

    sample_indices = alarm_internal_index + np.arange(
        0, internal_steps + 1, modeling_stride, dtype=np.int64
    )
    sample_clean = internal_clean[sample_indices].copy()
    sample_noisy = sample_clean.copy()
    sample_noisy[0] = alarm_observation
    if len(sample_noisy) > 1 and np.any(resolved_noise > 0):
        sample_noisy[1:] += np.random.default_rng(seed).normal(
            0.0, resolved_noise, size=sample_noisy[1:].shape
        )
    return PostAlarmSegment(
        alarm_internal_index=int(alarm_internal_index),
        internal_time=internal_time[alarm_internal_index : end_index + 1].copy(),
        internal_clean=internal_clean[alarm_internal_index : end_index + 1].copy(),
        sample_indices=sample_indices,
        sample_time=internal_time[sample_indices].copy(),
        sample_clean=sample_clean,
        sample_noisy=sample_noisy,
    )


def ecosystem_rhs(state: np.ndarray, parameters: EcosystemParameters) -> np.ndarray:
    x1, x2 = np.asarray(state, dtype=np.float64)
    dx1 = parameters.r * x1 * (1.0 - x1 / parameters.K) - parameters.alpha * x1 * x2
    dx2 = parameters.beta * parameters.alpha * x1 * x2 - parameters.d * x2
    return np.array([dx1, dx2], dtype=np.float64)


def euler_step(state: np.ndarray, dt: float, parameters: EcosystemParameters) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    state = np.asarray(state, dtype=np.float64)
    return np.maximum(state + dt * ecosystem_rhs(state, parameters), 0.0)


def simulate_sampled_dynamics(
    initial_state: np.ndarray,
    t_end: float,
    internal_dt: float,
    observation_stride: int,
    noise_std: float | np.ndarray,
    seed: int,
    step_function,
) -> SampledTrajectory:
    """Sample any fixed-step dynamical system on the detector time contract."""
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if initial_state.ndim != 1 or len(initial_state) < 1:
        raise ValueError("initial_state must be a non-empty one-dimensional array")
    if t_end <= 0 or internal_dt <= 0:
        raise ValueError("time settings must be positive")
    if not isinstance(observation_stride, (int, np.integer)) or observation_stride < 1:
        raise ValueError("observation_stride must be a positive integer")
    resolved_noise = resolve_state_noise(noise_std, len(initial_state))
    if (
        np.any(resolved_noise < 0)
    ):
        raise ValueError("noise_std must be non-negative and match the state dimension")

    steps = int(round(t_end / internal_dt))
    if not np.isclose(steps * internal_dt, t_end, rtol=0.0, atol=1e-12):
        raise ValueError("t_end must be an integer multiple of internal_dt")
    time = np.arange(steps + 1, dtype=np.float64) * internal_dt
    clean = np.empty((steps + 1, len(initial_state)), dtype=np.float64)
    clean[0] = initial_state
    for index in range(steps):
        next_state = np.asarray(
            step_function(clean[index], time[index], internal_dt), dtype=np.float64
        )
        if next_state.shape != initial_state.shape or not np.all(np.isfinite(next_state)):
            raise ValueError("step_function must return one finite value per state")
        clean[index + 1] = next_state

    sample_indices = np.arange(0, steps + 1, observation_stride, dtype=np.int64)
    sample_clean = clean[sample_indices].copy()
    sample_noisy = sample_clean + np.random.default_rng(seed).normal(
        0.0, resolved_noise, size=sample_clean.shape
    )
    return SampledTrajectory(
        internal_time=time,
        internal_clean=clean,
        sample_indices=sample_indices,
        sample_time=time[sample_indices],
        sample_clean=sample_clean,
        sample_noisy=sample_noisy,
    )


def simulate_sampled_trajectory(
    initial_state: np.ndarray,
    t_end: float,
    internal_dt: float,
    observation_stride: int,
    noise_std: float,
    seed: int,
    switch_time: float | None = None,
) -> SampledTrajectory:
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if internal_dt <= 0:
        raise ValueError("internal_dt must be positive")
    if not isinstance(observation_stride, (int, np.integer)) or observation_stride < 1:
        raise ValueError("observation_stride must be a positive integer")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if switch_time is not None and not 0.0 < switch_time < t_end:
        raise ValueError("switch_time must lie strictly inside (0, t_end)")

    def step(state, time, dt):
        parameters = (
            ABNORMAL_PARAMETERS
            if switch_time is not None and time >= switch_time
            else NORMAL_PARAMETERS
        )
        return euler_step(state, dt, parameters)

    return simulate_sampled_dynamics(
        initial_state=initial_state,
        t_end=t_end,
        internal_dt=internal_dt,
        observation_stride=observation_stride,
        noise_std=noise_std,
        seed=seed,
        step_function=step,
    )


def simulate_trajectory(
    initial_state: np.ndarray,
    t_end: float,
    dt: float,
    noise_std: float,
    seed: int,
    switch_time: float | None = None,
    observation_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = simulate_sampled_trajectory(
        initial_state=initial_state,
        t_end=t_end,
        internal_dt=dt,
        observation_stride=observation_stride,
        noise_std=noise_std,
        seed=seed,
        switch_time=switch_time,
    )
    return trajectory.sample_time, trajectory.sample_clean, trajectory.sample_noisy


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_normalization(trajectories: list[np.ndarray]) -> Normalization:
    if not trajectories:
        raise ValueError("at least one training trajectory is required")
    merged = np.concatenate([np.asarray(item, dtype=np.float64) for item in trajectories])
    if merged.ndim != 2 or merged.shape[1] < 1:
        raise ValueError("trajectories must have shape [time, channels]")
    channel_count = merged.shape[1]
    if any(np.asarray(item).ndim != 2 or np.asarray(item).shape[1] != channel_count for item in trajectories):
        raise ValueError("all trajectories must have the same channel count")
    return Normalization(mean=merged.mean(axis=0), std=np.maximum(merged.std(axis=0), 1e-8))


def make_windows(
    trajectory: np.ndarray,
    input_window: int,
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if input_window < 2:
        raise ValueError("input_window must be at least 2")
    if trajectory.ndim != 2 or trajectory.shape[1] < 1:
        raise ValueError("trajectory must have shape [time, channels]")
    if normalization.mean.shape != (trajectory.shape[1],) or normalization.std.shape != normalization.mean.shape:
        raise ValueError("normalization must match the trajectory channel count")
    if len(trajectory) <= input_window:
        raise ValueError("trajectory is too short for the selected input_window")

    normalized = normalization.transform(trajectory)
    inputs = np.stack(
        [normalized[index - input_window : index] for index in range(input_window, len(normalized))]
    )
    return inputs.astype(np.float32), normalized[input_window:].astype(np.float32)


def generate_normal_trajectories(
    count: int,
    duration: float,
    dt: float,
    noise_std: float,
    seed: int,
    observation_stride: int = 1,
) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = np.random.default_rng(seed)
    center = np.array([12.19, 24.4], dtype=np.float64)
    half_width = np.array([5.0, 8.0], dtype=np.float64)
    return [
        simulate_trajectory(
            initial_state=np.maximum(center + rng.uniform(-half_width, half_width), 0.5),
            t_end=duration,
            dt=dt,
            noise_std=noise_std,
            seed=seed + 1000 + index,
            switch_time=None,
            observation_stride=observation_stride,
        )[2]
        for index in range(count)
    ]


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class MGCRLSTM(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        conv_channels: int = 16,
        kernel_size: int = 3,
        hidden_size: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or conv_channels < 1 or hidden_size < 1 or lstm_layers < 1:
            raise ValueError("network dimensions must be positive")
        self.state_dim = state_dim
        self.causal_conv = CausalConv1d(state_dim, conv_channels, kernel_size)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_size, state_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != self.state_dim:
            raise ValueError("windows must have shape [batch, time, state_dim]")
        features = torch.tanh(self.causal_conv(windows.transpose(1, 2))).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        return self.output_head(torch.tanh(hidden[-1]))


@dataclass(frozen=True)
class GLRCalibration:
    residual_mean: np.ndarray
    residual_covariance: np.ndarray
    threshold: float
    window: int
    min_segment: int
    threshold_quantile: float
    covariance_regularization: float


@dataclass(frozen=True)
class GLRDetectionResult:
    energy: np.ndarray
    statistic: np.ndarray
    candidate_index: np.ndarray
    alarm: np.ndarray
    alarm_index: int | None
    estimated_switch_index: int | None


def fit_normal_residual_model(
    validation_residuals: list[np.ndarray], regularization: float
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("covariance regularization must be finite and positive")
    if not validation_residuals:
        raise ValueError("validation residuals cannot be empty")

    rows: list[np.ndarray] = []
    channel_count: int | None = None
    for residuals in validation_residuals:
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.ndim != 2:
            raise ValueError("residuals must have shape [time, channels]")
        if channel_count is None:
            channel_count = residuals.shape[1]
        elif residuals.shape[1] != channel_count:
            raise ValueError("validation residuals must have the same channel count")
        rows.append(residuals[np.isfinite(residuals).all(axis=1)])

    finite = np.concatenate(rows)
    if len(finite) < 2:
        raise ValueError("at least two finite validation residuals are required")
    mean = finite.mean(axis=0)
    covariance = np.atleast_2d(np.cov(finite, rowvar=False, bias=True))
    scale = float(np.trace(covariance)) / covariance.shape[0]
    if scale <= 0.0:
        raise ValueError("normal residual covariance has zero scale")
    return mean, covariance + regularization * scale * np.eye(covariance.shape[0])


def mahalanobis_energy(residuals: np.ndarray, calibration: GLRCalibration) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape [time, channels]")
    channels = residuals.shape[1]
    if calibration.residual_mean.shape != (channels,):
        raise ValueError("residual mean must contain one value per channel")
    if calibration.residual_covariance.shape != (channels, channels):
        raise ValueError("residual covariance must match the channel count")

    energy = np.full(len(residuals), np.nan, dtype=np.float64)
    finite = np.isfinite(residuals).all(axis=1)
    centered = residuals[finite] - calibration.residual_mean
    solved = np.linalg.solve(calibration.residual_covariance, centered.T).T
    energy[finite] = np.einsum("ij,ij->i", centered, solved)
    return energy


def _segment_glr(total: float, length: int, channels: int) -> float:
    rho = max(1.0, total / (length * channels))
    return 0.5 * length * channels * (rho - 1.0 - np.log(rho))


def window_limited_glr(
    energy: np.ndarray,
    channels: int,
    window: int,
    min_segment: int,
    stop_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1:
        raise ValueError("energy must be one-dimensional")
    if channels < 1:
        raise ValueError("channels must be positive")
    if min_segment < 2:
        raise ValueError("minimum GLR segment must be at least 2")
    if window < min_segment:
        raise ValueError("GLR window must be at least the minimum segment")
    if stop_threshold is not None and stop_threshold < 0.0:
        raise ValueError("GLR stop threshold cannot be negative")

    statistic = np.full(len(energy), np.nan, dtype=np.float64)
    candidate_index = np.full(len(energy), -1, dtype=np.int64)
    finite = np.isfinite(energy)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(np.where(finite, energy, 0.0))))
    cumulative_finite = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))

    for end in range(min_segment - 1, len(energy)):
        first_candidate = max(0, end - window + 1)
        last_candidate = end - min_segment + 1
        best_score, best_candidate = -np.inf, -1
        for start in range(first_candidate, last_candidate + 1):
            length = end - start + 1
            if cumulative_finite[end + 1] - cumulative_finite[start] != length:
                continue
            total = cumulative_energy[end + 1] - cumulative_energy[start]
            score = _segment_glr(float(total), length, channels)
            if score > best_score:
                best_score, best_candidate = score, start
        if best_candidate < 0:
            continue
        statistic[end], candidate_index[end] = best_score, best_candidate
        if stop_threshold is not None and best_score > stop_threshold:
            return statistic, candidate_index, end, best_candidate
    return statistic, candidate_index, None, None


def calibrate_glr(
    validation_residuals: list[np.ndarray],
    window: int,
    min_segment: int,
    quantile: float,
    covariance_regularization: float,
) -> GLRCalibration:
    if min_segment < 2:
        raise ValueError("minimum GLR segment must be at least 2")
    if window < min_segment:
        raise ValueError("GLR window must be at least the minimum segment")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("GLR threshold quantile must lie inside (0, 1]")

    mean, covariance = fit_normal_residual_model(validation_residuals, covariance_regularization)
    provisional = GLRCalibration(
        mean, covariance, np.nan, window, min_segment, quantile, covariance_regularization
    )
    normal_statistics = []
    for residuals in validation_residuals:
        statistic, _, _, _ = window_limited_glr(
            mahalanobis_energy(residuals, provisional), mean.size, window, min_segment
        )
        normal_statistics.append(statistic[np.isfinite(statistic)])
    finite_statistics = np.concatenate(normal_statistics)
    if len(finite_statistics) == 0:
        raise ValueError("validation trajectories are too short for GLR calibration")
    return GLRCalibration(
        mean,
        covariance,
        float(np.quantile(finite_statistics, quantile)),
        window,
        min_segment,
        quantile,
        covariance_regularization,
    )


def detect_switch(residuals: np.ndarray, calibration: GLRCalibration) -> GLRDetectionResult:
    residuals = np.asarray(residuals, dtype=np.float64)
    energy = mahalanobis_energy(residuals, calibration)
    statistic, candidate_index, alarm_index, estimated_index = window_limited_glr(
        energy,
        channels=residuals.shape[1],
        window=calibration.window,
        min_segment=calibration.min_segment,
        stop_threshold=calibration.threshold,
    )
    alarm = np.zeros(len(residuals), dtype=bool)
    if alarm_index is not None:
        alarm[alarm_index] = True
    return GLRDetectionResult(
        energy, statistic, candidate_index, alarm, alarm_index, estimated_index
    )
