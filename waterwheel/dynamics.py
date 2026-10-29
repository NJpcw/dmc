"""Physical waterwheel model and multirate noisy simulation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from common.detection import residual_glr_detection as CORE


STATE_DIM = 3
Q_BEFORE = 18.0
Q_AFTER = 15.0
SWITCH_TIME = 100.0
DEFAULT_INITIAL_STATE = np.array([4.5, 3.7, 1.2], dtype=float)
DEFAULT_NOISE_STD = np.array([0.03, 0.03, 0.01], dtype=float)

SampledTrajectory = CORE.SampledTrajectory


def _state_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (STATE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def resolve_noise_std(noise_std: np.ndarray | float) -> np.ndarray:
    values = np.asarray(noise_std, dtype=float)
    if values.ndim == 0:
        values = np.full(STATE_DIM, float(values))
    if values.shape != (STATE_DIM,):
        raise ValueError("noise_std must be scalar or contain one value per state")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("noise_std values must be finite and non-negative")
    return values


def _scalar_input(value: float, name: str) -> float:
    values = np.asarray(value, dtype=float)
    if values.ndim != 0 or not np.isfinite(values):
        raise ValueError(f"{name} must be a finite scalar")
    return float(values)


def waterwheel_rhs(
    state: np.ndarray,
    q: float,
    control: float = 0.0,
    disturbance: float = 0.0,
) -> np.ndarray:
    """Evaluate the controlled Malkus--Lorenz waterwheel vector field."""
    omega, a, b = _state_vector(state, "state")
    if not np.isfinite(q):
        raise ValueError("q must be finite")
    matched_input = _scalar_input(control, "control") + _scalar_input(disturbance, 'disturbance')
    return np.array(
        [-3.0 * omega + 3.0 * a + matched_input, omega * b - a, -omega * a - b + float(q)],
        dtype=float,
    )


def euler_step(
    state: np.ndarray,
    dt: float,
    q: float,
    control: float = 0.0,
    disturbance: float = 0.0,
) -> np.ndarray:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    state = _state_vector(state, "state")
    return state + dt * waterwheel_rhs(state, q=q, control=control, disturbance=disturbance)


def simulate_sampled_trajectory(
    initial_state: np.ndarray = DEFAULT_INITIAL_STATE,
    t_end: float = 110.0,
    internal_dt: float = 0.001,
    observation_stride: int = 50,
    noise_std: np.ndarray | float = DEFAULT_NOISE_STD,
    seed: int = 2026,
    switch_time: float | None = SWITCH_TIME,
    q_before: float = Q_BEFORE,
    q_after: float = Q_AFTER,
    control_function: Callable[[np.ndarray, float], float] | None = None,
) -> SampledTrajectory:
    """Simulate one continuous record and sample noisy observations from it."""
    initial_state = _state_vector(initial_state, "initial_state")
    state_noise = resolve_noise_std(noise_std)
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive")
    if not np.isfinite(internal_dt) or internal_dt <= 0.0:
        raise ValueError("internal_dt must be finite and positive")
    if not isinstance(observation_stride, (int, np.integer)) or observation_stride < 1:
        raise ValueError("observation_stride must be a positive integer")
    if switch_time is not None and not 0.0 < switch_time < t_end:
        raise ValueError("switch_time must lie strictly inside (0, t_end)")

    def step_function(state: np.ndarray, time: float, dt: float) -> np.ndarray:
        q = q_after if switch_time is not None and time >= switch_time else q_before
        control = 0.0 if control_function is None else _scalar_input(
            control_function(state.copy(), time), "control"
        )
        return euler_step(state, dt, q=q, control=control)

    return CORE.simulate_sampled_dynamics(
        initial_state=initial_state,
        t_end=t_end,
        internal_dt=internal_dt,
        observation_stride=observation_stride,
        noise_std=state_noise,
        seed=seed,
        step_function=step_function,
    )


def generate_normal_trajectories(
    count: int,
    duration: float,
    internal_dt: float,
    observation_stride: int,
    noise_std: np.ndarray | float,
    seed: int,
) -> list[np.ndarray]:
    """Generate normal-regime records for detector training and calibration."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    center = DEFAULT_INITIAL_STATE
    half_width = np.array([0.8, 0.8, 0.2], dtype=float)
    trajectories = []
    for index in range(count):
        initial_state = center + rng.uniform(-half_width, half_width)
        trajectory = simulate_sampled_trajectory(
            initial_state=initial_state,
            t_end=duration,
            internal_dt=internal_dt,
            observation_stride=observation_stride,
            noise_std=noise_std,
            seed=seed + 1000 + index,
            switch_time=None,
        )
        trajectories.append(trajectory.sample_noisy)
    return trajectories
