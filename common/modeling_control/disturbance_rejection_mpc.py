from dataclasses import dataclass, field
from pathlib import Path
import time
from types import SimpleNamespace

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import LinearConstraint

from . import ngrc_modeling as ROBUST_NGRC


# 2. Closed-loop configuration


@dataclass
class ControlConfig:
    model_key: str = "bli"
    reference: np.ndarray = field(default_factory=lambda: np.array([9.38, 20.31], dtype=float))
    control_duration: float = 6.0
    horizon: int = 20
    move_horizon: int | None = None
    internal_dt: float = 0.001
    control_stride: int = 10
    input_matrix: np.ndarray | None = None
    state_lower_bound: np.ndarray | float | None = 1e-8

    measurement_noise_std: np.ndarray | float = 0.05
    observer_bandwidth: float = 10.0
    with_disturbance: bool = True
    random_seed: int = 10

    q_state: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0], dtype=float))
    r_control: float = 0.1
    r_delta: float = 0.1
    terminal_weight: float = 5.0

    u_min: np.ndarray = field(default_factory=lambda: np.array([-15.0, -15.0], dtype=float))
    u_max: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0], dtype=float))
    du_max: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5], dtype=float))

    mpc_maxiter: int = 100
    mpc_ftol: float = 1e-4
    prediction_failure_cost: float = 1e12
    steady_state_window: float = 2.0

    def __post_init__(self):
        if self.move_horizon is None:
            self.move_horizon = min(10, self.horizon)

    def resolved_input_matrix(self):
        state_dim = len(np.asarray(self.reference, dtype=float))
        if self.input_matrix is None:
            return np.eye(state_dim)
        matrix = np.asarray(self.input_matrix, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != state_dim
            or matrix.shape[1] < 1
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError('input_matrix must be a finite state_dim-by-input_dim matrix.')
        return matrix

    def resolved_noise_std(self):
        state_dim = len(np.asarray(self.reference, dtype=float))
        noise = np.asarray(self.measurement_noise_std, dtype=float)
        if noise.ndim == 0:
            noise = np.full(state_dim, float(noise))
        if noise.shape != (state_dim,) or not np.all(np.isfinite(noise)):
            raise ValueError('measurement_noise_std must be scalar or a finite state vector.')
        return noise

    def resolved_state_lower_bound(self):
        if self.state_lower_bound is None:
            return None
        state_dim = len(np.asarray(self.reference, dtype=float))
        lower = np.asarray(self.state_lower_bound, dtype=float)
        if lower.ndim == 0:
            lower = np.full(state_dim, float(lower))
        if lower.shape != (state_dim,) or not np.all(np.isfinite(lower)):
            raise ValueError('state_lower_bound must be scalar, a finite state vector, or None.')
        return lower

    def validate(self, dt):
        state_arrays = {'reference': self.reference, 'q_state': self.q_state}
        input_arrays = {'u_min': self.u_min, 'u_max': self.u_max, 'du_max': self.du_max}
        converted = {}
        for name, value in {**state_arrays, **input_arrays}.items():
            array = np.asarray(value, dtype=float)
            if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a non-empty finite 1-D array.")
            converted[name] = array

        dim = len(converted["reference"])
        if any(len(converted[name]) != dim for name in state_arrays):
            raise ValueError("All state parameter arrays must match reference length.")
        input_dim = self.resolved_input_matrix().shape[1]
        if any(len(converted[name]) != input_dim for name in input_arrays):
            raise ValueError("All input parameter arrays must match input_matrix columns.")
        noise_std = self.resolved_noise_std()
        self.resolved_state_lower_bound()
        if np.any(converted["u_min"] >= converted["u_max"]):
            raise ValueError("Every u_min entry must be smaller than u_max.")
        if np.any(converted["du_max"] <= 0) or np.any(converted["q_state"] < 0):
            raise ValueError("du_max must be positive and q_state cannot be negative.")
        scalar_values = (
            self.control_duration,
            self.internal_dt,
            self.observer_bandwidth,
            self.r_control,
            self.r_delta,
            self.terminal_weight,
            self.mpc_ftol,
            self.prediction_failure_cost,
            self.steady_state_window,
        )
        if not np.all(np.isfinite(np.asarray(scalar_values, dtype=float))):
            raise ValueError("Every scalar control parameter must be finite.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("The control sampling interval must be positive.")
        if self.control_duration <= 0 or int(round(self.control_duration / dt)) < 1:
            raise ValueError("control_duration must contain at least one control interval.")
        if self.internal_dt <= 0:
            raise ValueError("internal_dt must be positive.")
        if not isinstance(self.control_stride, (int, np.integer)) or self.control_stride <= 0:
            raise ValueError("control_stride must be a positive integer.")
        if not np.isclose(self.internal_dt * self.control_stride, dt, rtol=0.0, atol=1e-12):
            raise ValueError("internal_dt times control_stride must equal the control interval.")
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if (
            not isinstance(self.move_horizon, (int, np.integer))
            or self.move_horizon <= 0
            or self.move_horizon > self.horizon
        ):
            raise ValueError("move_horizon must be an integer inside [1, horizon].")
        if np.any(noise_std < 0) or self.observer_bandwidth <= 0:
            raise ValueError("Noise standard deviation cannot be negative and ESO bandwidth must be positive.")
        if self.r_control < 0 or self.r_delta < 0 or self.terminal_weight < 0:
            raise ValueError("MPC cost weights cannot be negative.")
        if self.mpc_maxiter <= 0 or self.mpc_ftol <= 0:
            raise ValueError("MPC solver settings must be positive.")
        if self.prediction_failure_cost <= 0 or self.steady_state_window <= 0:
            raise ValueError('Failure cost and final steady-state window must be positive.')


# 3. Controlled plant and unknown disturbance


def unknown_total_disturbance(x, t):
    x = np.asarray(x, dtype=float)
    external = np.array([0.25 * np.sin(1.2 * t) + 0.1 * np.sin(0.25 * t), -0.2 * np.cos(0.9 * t)])
    mismatch = np.array([0.01 * x[0] * np.sin(0.4 * t), -0.008 * x[1] * np.cos(0.35 * t)])
    disturbance_scale = 4.0
    return disturbance_scale * (external + mismatch)


def controlled_grassland_rhs(x, t, u, with_disturbance=True):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    nominal = np.asarray(ROBUST_NGRC.grassland_abnormal_system(x, t), dtype=float)
    disturbance = unknown_total_disturbance(x, t) if with_disturbance else 0.0
    return nominal + u + disturbance


def euler_step_controlled_plant(x, t, u, dt, with_disturbance=True):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    rate = controlled_grassland_rhs(x, t, u, with_disturbance=with_disturbance)
    x_next = x + dt * rate
    return np.maximum(x_next, 1e-8)


def euler_step_controlled_plant_substeps(
    x,
    t,
    u,
    internal_dt=0.001,
    substeps=10,
    with_disturbance=True,
):
    if internal_dt <= 0:
        raise ValueError("internal_dt must be positive.")
    if not isinstance(substeps, (int, np.integer)) or substeps <= 0:
        raise ValueError("substeps must be a positive integer.")
    state = np.asarray(x, dtype=float).copy()
    time = float(t)
    for _ in range(int(substeps)):
        state = euler_step_controlled_plant(
            state,
            time,
            u,
            internal_dt,
            with_disturbance=with_disturbance,
        )
        time += internal_dt
    return state


# 4. Robust NGRC controlled prediction


def minimum_history_length(model):
    return (int(model["k"]) - 1) * int(model["s"]) + 1


def append_and_trim(history, state, max_length):
    history = np.vstack([np.asarray(history, dtype=float), np.asarray(state, dtype=float)])
    return history[-max_length:]


def ngrc_one_step_nominal(model, history_original):
    history = np.asarray(history_original, dtype=float)
    required = minimum_history_length(model)
    if history.ndim != 2 or len(history) < required:
        raise ValueError("Insufficient two-dimensional history for NGRC prediction.")

    return ROBUST_NGRC.predict_one_step_original(model, history[-required:])


def ngrc_one_step_with_rate_input(model, history_original, rate_input, dt):
    """Apply BLI rate inputs in the continuous-rate Euler prediction path."""
    history = np.asarray(history_original, dtype=float)
    rate_input = np.asarray(rate_input, dtype=float)
    if _is_bli_model(model):
        scaler = model["scaler"]
        state_scaled = scaler.transform(history[-1:].copy())[0]
        next_scaled = ROBUST_NGRC._BLI.euler_step_scaled(
            model["W_out"],
            model["feature_transformer"],
            state_scaled,
            dt,
            additive_scaled=rate_input / scaler.scale_,
        )
        return scaler.inverse_transform(next_scaled.reshape(1, -1))[0]
    return ngrc_one_step_nominal(model, history) + dt * rate_input


def _is_bli_model(model):
    return model.get("method") in ("bli", "bli_delay")


def _casadi_scaler_parameters(model):
    scaler = model["scaler"]
    if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
        center = np.asarray(scaler.center_, dtype=float)
        scale = np.asarray(scaler.scale_, dtype=float)
    else:
        center = np.asarray(scaler.mean_, dtype=float)
        scale = np.asarray(scaler.std_, dtype=float)
    if center.ndim != 1 or scale.shape != center.shape or np.any(scale <= 0):
        raise ValueError("NGRC scaler parameters must be matching finite vectors.")
    return center, scale


def _casadi_guarded_quadratic_features(model, scaled):
    transformer = model["feature_transformer"]
    training_limit = np.asarray(transformer.training_limit_, dtype=float)
    exact_limit = training_limit * (1.0 + float(transformer.extrapolation_margin))
    dim = int(transformer.state_dim_)
    if scaled.numel() != dim or training_limit.shape != (dim,):
        raise ValueError("Symbolic state and fitted feature dimensions must agree.")
    guarded = []
    for channel in range(dim):
        value = scaled[channel]
        magnitude = ca.fabs(value)
        limit = float(exact_limit[channel])
        bounded = limit + limit * ca.tanh((magnitude - limit) / limit)
        guarded.append(ca.if_else(magnitude <= limit, value, ca.sign(value) * bounded))
    values = [ca.MX(1.0), *guarded]
    values.extend(guarded[left] * guarded[right] for left, right in transformer.pairs_)
    return ca.vertcat(*values)


def _casadi_bli_next_state(model, current_state, rate_input, dt):
    center, scale = _casadi_scaler_parameters(model)
    scaled = (current_state - ca.DM(center)) / ca.DM(scale)
    features = _casadi_guarded_quadratic_features(model, scaled)
    scaled_rate = ca.DM(np.asarray(model["W_out"], dtype=float)) @ features
    scaled_rate += rate_input / ca.DM(scale)
    next_scaled = scaled + float(dt) * scaled_rate
    return next_scaled * ca.DM(scale) + ca.DM(center)


def _casadi_raw_next_state(model, history, rate_input, dt):
    center, scale = _casadi_scaler_parameters(model)
    dim = len(center)
    k = int(model["k"])
    stride = int(model["s"])
    required = minimum_history_length(model)
    scaled_rows = [
        (history[row, :].T - ca.DM(center)) / ca.DM(scale)
        for row in range(required)
    ]
    current = required - 1
    linear = ca.vertcat(*[scaled_rows[current - delay * stride] for delay in range(k)])
    if "feature_transformer" in model:
        features = _casadi_guarded_quadratic_features(model, linear)
        increment = ca.DM(np.asarray(model["W_out"], dtype=float)) @ features
        nominal_next = (scaled_rows[-1] + increment) * ca.DM(scale) + ca.DM(center)
        if rate_input.numel() != dim:
            raise ValueError("Rate input and Raw-NGRC state dimensions must agree.")
        return nominal_next + float(dt) * rate_input

    feature_clip = float(model["feature_clip"])
    linear = ca.fmin(ca.fmax(linear, -feature_clip), feature_clip)
    feature_values = [ca.MX(1.0)]
    feature_values.extend(linear[index] for index in range(linear.numel()))
    poly_order = int(model["poly_order"])
    if poly_order >= 2:
        for first in range(linear.numel()):
            for second in range(first, linear.numel()):
                feature_values.append(linear[first] * linear[second])
    if poly_order >= 3:
        for first in range(linear.numel()):
            for second in range(first, linear.numel()):
                for third in range(second, linear.numel()):
                    feature_values.append(linear[first] * linear[second] * linear[third])
    features = ca.vertcat(*feature_values)
    increment = ca.DM(np.asarray(model["W_out"], dtype=float)) @ features
    nominal_next = (scaled_rows[-1] + increment) * ca.DM(scale) + ca.DM(center)
    if rate_input.numel() != dim:
        raise ValueError("Rate input and Raw-NGRC state dimensions must agree.")
    return nominal_next + float(dt) * rate_input


def build_casadi_ngrc_step_function(model, dt):
    """Build an exact symbolic counterpart of the NumPy controlled NGRC step."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    required = minimum_history_length(model)
    center, _ = _casadi_scaler_parameters(model)
    dim = len(center)
    history = ca.MX.sym("history", required, dim)
    rate_input = ca.MX.sym("rate_input", dim)
    if _is_bli_model(model):
        next_state = _casadi_bli_next_state(model, history[required - 1, :].T, rate_input, dt)
    elif model.get("method") in (None, "raw", "raw_delay"):
        next_state = _casadi_raw_next_state(model, history, rate_input, dt)
    else:
        raise ValueError("CasADi MPC supports only Raw-NGRC and BLI-NGRC models.")
    return ca.Function(
        "ngrc_controlled_step",
        [history, rate_input],
        [next_state],
        ["history", "rate_input"],
        ["next_state"],
    )


def rollout_ngrc_controlled(
    model,
    history_original,
    controls,
    dt,
    disturbance=None,
    max_abs_state=1e3,
    state_lower_bound=1e-8,
):
    controls = np.asarray(controls, dtype=float)
    history = np.asarray(history_original, dtype=float)
    if controls.ndim != 2 or history.ndim != 2:
        raise ValueError("history_original and controls must both be 2-D arrays.")
    if controls.shape[1] != history.shape[1]:
        raise ValueError("Control and state dimensions must agree.")
    if dt <= 0 or max_abs_state <= 0:
        raise ValueError("dt and max_abs_state must be positive.")

    dim = history.shape[1]
    if state_lower_bound is not None:
        lower = np.asarray(state_lower_bound, dtype=float)
        if lower.ndim == 0:
            lower = np.full(dim, float(lower))
        if lower.shape != (dim,) or not np.all(np.isfinite(lower)):
            raise ValueError('state_lower_bound must be scalar, a state vector, or None.')
        state_lower_bound = lower
    if disturbance is None:
        disturbance = np.zeros(dim)
    disturbance = np.asarray(disturbance, dtype=float)
    if disturbance.shape != (dim,):
        raise ValueError("disturbance must have one entry per state channel.")

    required = minimum_history_length(model)
    if len(history) < required:
        raise ValueError("Insufficient history for the selected NGRC delays.")
    rolling_history = history[-required:].copy()
    prediction = [rolling_history[-1].copy()]

    for control in controls:
        next_state = ngrc_one_step_with_rate_input(
            model, rolling_history, control + disturbance, dt
        )
        if state_lower_bound is not None:
            next_state = np.maximum(next_state, state_lower_bound)
        if not np.all(np.isfinite(next_state)) or np.max(np.abs(next_state)) > max_abs_state:
            missing = len(controls) + 1 - len(prediction)
            prediction.extend([np.full(dim, np.nan) for _ in range(missing)])
            break
        prediction.append(next_state.copy())
        rolling_history = append_and_trim(rolling_history, next_state, required)

    return np.asarray(prediction)


# 5. Time-aligned NGRC residual ESO


class NGRCResidualESO:
    def __init__(self, dim=2, observer_bandwidth=8.0, input_matrix=None, state_lower_bound=1e-08):
        if dim <= 0 or observer_bandwidth <= 0:
            raise ValueError("ESO dimension and bandwidth must be positive.")
        self.dim = int(dim)
        self.omega = float(observer_bandwidth)
        self.beta1 = 2.0 * self.omega
        self.beta2 = self.omega**2
        if input_matrix is None:
            input_matrix = np.eye(self.dim)
        self.input_matrix = np.asarray(input_matrix, dtype=float)
        if (
            self.input_matrix.ndim != 2
            or self.input_matrix.shape[0] != self.dim
            or self.input_matrix.shape[1] < 1
            or not np.all(np.isfinite(self.input_matrix))
        ):
            raise ValueError('ESO input_matrix must be a finite dim-by-input_dim matrix.')
        self.input_dim = self.input_matrix.shape[1]
        if state_lower_bound is None:
            self.state_lower_bound = None
        else:
            lower = np.asarray(state_lower_bound, dtype=float)
            if lower.ndim == 0:
                lower = np.full(self.dim, float(lower))
            if lower.shape != (self.dim,) or not np.all(np.isfinite(lower)):
                raise ValueError('ESO state_lower_bound must be scalar, a state vector, or None.')
            self.state_lower_bound = lower
        self.z1 = None
        self.z2 = None
        self.update_count = 0

    def reset(self, measurement):
        measurement = np.asarray(measurement, dtype=float)
        if measurement.shape != (self.dim,):
            raise ValueError("ESO measurement has the wrong dimension.")
        self.z1 = measurement.copy()
        self.z2 = np.zeros(self.dim)
        self.update_count = 0

    def update(self, measurement, previous_u, dt, model, observer_history, propagate=True):
        measurement = np.asarray(measurement, dtype=float)
        previous_u = np.asarray(previous_u, dtype=float)
        if measurement.shape != (self.dim,) or previous_u.shape != (self.input_dim,):
            raise ValueError("ESO measurement or physical-input dimension is invalid.")
        if dt <= 0:
            raise ValueError("ESO sampling interval must be positive.")

        if self.z1 is None:
            self.reset(measurement)
            self.update_count = 1
            return self.z1.copy(), self.z2.copy()

        if propagate:
            history = np.asarray(observer_history, dtype=float).copy()
            required = minimum_history_length(model)
            if history.ndim != 2 or len(history) < required:
                raise ValueError("Observer history is too short for NGRC propagation.")
            history = history[-required:]
            history[-1] = self.z1.copy()
            nominal_next = ngrc_one_step_nominal(model, history)
            self.z1 = nominal_next + dt * (
                self.input_matrix @ previous_u + self.z2
            )
            if self.state_lower_bound is not None:
                self.z1 = np.maximum(self.z1, self.state_lower_bound)

        innovation = measurement - self.z1
        self.z1 = self.z1 + dt * self.beta1 * innovation
        self.z2 = self.z2 + dt * self.beta2 * innovation
        if self.state_lower_bound is not None:
            self.z1 = np.maximum(self.z1, self.state_lower_bound)
        self.update_count += 1
        return self.z1.copy(), self.z2.copy()


class BoundedResidualDisturbanceObserver:
    """Robustly smooth and bound the NGRC one-step residual disturbance."""

    def __init__(self, dim, smoothing_gain, huber_threshold, magnitude_limit, slew_limit):
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("Observer dimension must be a positive integer.")
        if (
            not np.isfinite(smoothing_gain)
            or smoothing_gain <= 0
            or smoothing_gain > 1
        ):
            raise ValueError("smoothing_gain must lie in (0, 1].")
        self.dim = int(dim)
        self.smoothing_gain = float(smoothing_gain)
        self.huber_threshold = self._positive_vector(huber_threshold, 'huber_threshold')
        self.magnitude_limit = self._positive_vector(magnitude_limit, 'magnitude_limit')
        self.slew_limit = self._positive_vector(slew_limit, "slew_limit")
        self.d_hat = np.zeros(self.dim)
        self.update_count = 0
        self.magnitude_saturation_count = 0
        self.slew_saturation_count = 0

    def _positive_vector(self, value, name):
        vector = np.asarray(value, dtype=float)
        if (
            vector.shape != (self.dim,)
            or not np.all(np.isfinite(vector))
            or np.any(vector <= 0)
        ):
            raise ValueError(f"{name} must be a positive finite dim-vector.")
        return vector

    def update(self, current_state, nominal_next, previous_u, dt):
        current_state = np.asarray(current_state, dtype=float)
        nominal_next = np.asarray(nominal_next, dtype=float)
        previous_u = np.asarray(previous_u, dtype=float)
        if any(
            vector.shape != (self.dim,) or not np.all(np.isfinite(vector))
            for vector in (current_state, nominal_next, previous_u)
        ):
            raise ValueError("Observer states and input must be finite dim-vectors.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Observer dt must be finite and positive.")

        raw_residual = (current_state - nominal_next) / dt - previous_u
        innovation = np.clip(raw_residual - self.d_hat, -self.huber_threshold, self.huber_threshold)
        candidate = self.d_hat + self.smoothing_gain * innovation
        requested_delta = candidate - self.d_hat
        limited_delta = np.clip(requested_delta, -self.slew_limit, self.slew_limit)
        self.slew_saturation_count += int(
            np.count_nonzero(np.abs(requested_delta) > self.slew_limit)
        )
        candidate = self.d_hat + limited_delta
        bounded = np.clip(candidate, -self.magnitude_limit, self.magnitude_limit)
        self.magnitude_saturation_count += int(
            np.count_nonzero(np.abs(candidate) > self.magnitude_limit)
        )
        self.d_hat = bounded
        self.update_count += 1
        return self.d_hat.copy(), raw_residual.copy()


# 6. Constrained warm-start MPC in physical input coordinates


def project_control_sequence(sequence, previous_u, u_min, u_max, du_max):
    sequence = np.asarray(sequence, dtype=float).copy()
    previous = np.asarray(previous_u, dtype=float).copy()
    for index in range(len(sequence)):
        candidate = np.clip(sequence[index], u_min, u_max)
        candidate = previous + np.clip(candidate - previous, -du_max, du_max)
        candidate = np.clip(candidate, u_min, u_max)
        sequence[index] = candidate
        previous = candidate
    return sequence


def build_virtual_slew_linear_constraint(previous_u, disturbance_hat, du_max, move_horizon):
    """Build exact linear slew constraints for virtual inputs v = u + d_hat."""
    previous_u = np.asarray(previous_u, dtype=float)
    disturbance_hat = np.asarray(disturbance_hat, dtype=float)
    du_max = np.asarray(du_max, dtype=float)
    if previous_u.ndim != 1 or previous_u.shape != disturbance_hat.shape:
        raise ValueError("previous_u and disturbance_hat must be equal 1-D vectors.")
    if du_max.shape != previous_u.shape or np.any(du_max <= 0):
        raise ValueError("du_max must be a positive vector matching previous_u.")
    if not isinstance(move_horizon, (int, np.integer)) or move_horizon <= 0:
        raise ValueError("move_horizon must be a positive integer.")

    dim = len(previous_u)
    count = int(move_horizon) * dim
    matrix = np.zeros((count, count), dtype=float)
    lower = np.empty(count, dtype=float)
    upper = np.empty(count, dtype=float)
    for move in range(int(move_horizon)):
        for channel in range(dim):
            row = move * dim + channel
            matrix[row, row] = 1.0
            if move == 0:
                lower[row] = (
                    previous_u[channel]
                    + disturbance_hat[channel]
                    - du_max[channel]
                )
                upper[row] = (
                    previous_u[channel]
                    + disturbance_hat[channel]
                    + du_max[channel]
                )
            else:
                matrix[row, (move - 1) * dim + channel] = -1.0
                lower[row] = -du_max[channel]
                upper[row] = du_max[channel]
    return LinearConstraint(matrix, lower, upper)


def _mpc_cache_key(
    model,
    horizon,
    move_horizon,
    dt,
    q_state,
    r_control,
    r_delta,
    terminal_weight,
    du_max,
    maxiter,
    ftol,
    failure_cost,
    input_matrix,
    state_lower_bound,
):
    transformer = model.get("feature_transformer")
    scaler_center, scaler_scale = _casadi_scaler_parameters(model)
    transformer_signature = ()
    if _is_bli_model(model):
        transformer_signature = (
            tuple(np.asarray(transformer.training_limit_, dtype=float)),
            float(transformer.extrapolation_margin),
        )
    return (
        model.get("method", "raw"),
        int(model["k"]),
        int(model["s"]),
        int(model["poly_order"]),
        float(model["feature_clip"]),
        tuple(scaler_center),
        tuple(scaler_scale),
        tuple(np.asarray(model["W_out"]).shape),
        np.asarray(model["W_out"], dtype=float).tobytes(),
        transformer_signature,
        int(horizon),
        int(move_horizon),
        float(dt),
        tuple(np.asarray(q_state, dtype=float)),
        float(r_control),
        float(r_delta),
        float(terminal_weight),
        tuple(np.asarray(du_max, dtype=float)),
        int(maxiter),
        float(ftol),
        float(failure_cost),
        np.asarray(input_matrix, dtype=float).tobytes(),
        (
            None if state_lower_bound is None else tuple(np.asarray(state_lower_bound, dtype=float))
        ),
    )


def _build_casadi_mpc_problem(
    model,
    horizon,
    move_horizon,
    dt,
    q_state,
    r_control,
    r_delta,
    terminal_weight,
    du_max,
    maxiter,
    ftol,
    failure_cost,
    input_matrix,
    state_lower_bound,
    cache_key,
):
    required = minimum_history_length(model)
    state_dim = len(np.asarray(q_state, dtype=float))
    input_matrix = np.asarray(input_matrix, dtype=float)
    input_dim = input_matrix.shape[1]
    decision_count = move_horizon * input_dim
    parameter_count = required * state_dim + state_dim + 2 * input_dim
    decisions = ca.MX.sym("virtual_moves", decision_count)
    parameters = ca.MX.sym("mpc_parameters", parameter_count)

    offset = 0
    history = []
    for _ in range(required):
        history.append(parameters[offset : offset + state_dim])
        offset += state_dim
    reference = parameters[offset : offset + state_dim]
    offset += state_dim
    previous_u = parameters[offset : offset + input_dim]
    offset += input_dim
    disturbance_hat = parameters[offset : offset + input_dim]

    moves = [
        decisions[index * input_dim : (index + 1) * input_dim]
        for index in range(move_horizon)
    ]
    virtual_controls = moves + [moves[-1]] * (horizon - move_horizon)
    actual_controls = [control - disturbance_hat for control in virtual_controls]

    objective = ca.MX(0.0)
    constraints = []
    previous_actual = previous_u
    for move_index in range(move_horizon):
        current_actual = moves[move_index] - disturbance_hat
        constraints.append(current_actual - previous_actual)
        previous_actual = current_actual

    q_weights = ca.DM(np.asarray(q_state, dtype=float))
    symbolic_input_matrix = ca.DM(input_matrix)
    rolling_history = list(history)
    prediction_valid = ca.MX(1.0)
    previous_actual = previous_u
    for step in range(horizon):
        history_matrix = ca.vertcat(*[state.T for state in rolling_history])
        state_rate_input = symbolic_input_matrix @ virtual_controls[step]
        if _is_bli_model(model):
            next_state = _casadi_bli_next_state(model, rolling_history[-1], state_rate_input, dt)
        else:
            next_state = _casadi_raw_next_state(model, history_matrix, state_rate_input, dt)
        if state_lower_bound is not None:
            next_state = ca.fmax(next_state, ca.DM(np.asarray(state_lower_bound, dtype=float)))
        prediction_valid *= ca.if_else(ca.mmax(ca.fabs(next_state)) <= 1e3, 1.0, 0.0)
        error = next_state - reference
        actual = actual_controls[step]
        increment = actual - previous_actual
        objective += ca.dot(q_weights, error * error)
        objective += float(r_control) * ca.dot(actual, actual)
        objective += float(r_delta) * ca.dot(increment, increment)
        previous_actual = actual
        rolling_history = (rolling_history + [next_state])[-required:]

    terminal_error = rolling_history[-1] - reference
    objective += float(terminal_weight) * ca.dot(q_weights, terminal_error * terminal_error)
    objective = ca.if_else(prediction_valid > 0.5, objective, float(failure_cost))
    constraint_vector = ca.vertcat(*constraints)
    nlp = {"x": decisions, "p": parameters, "f": objective, "g": constraint_vector}
    solver_options = {
        "expand": True,
        "print_time": False,
        "ipopt.sb": "yes",
        "ipopt.print_level": 0,
        "ipopt.max_iter": int(maxiter),
        "ipopt.tol": float(ftol),
        "ipopt.warm_start_init_point": "yes",
    }
    solver_name = f"ngrc_mpc_{abs(hash(cache_key))}"
    solver = ca.nlpsol(solver_name, "ipopt", nlp, solver_options)
    return {
        "solver": solver,
        "lbg": np.tile(-np.asarray(du_max, dtype=float), move_horizon),
        "ubg": np.tile(np.asarray(du_max, dtype=float), move_horizon),
        "lam_x": None,
        "lam_g": None,
    }


def _get_casadi_mpc_problem(model, cache_key, **builder_arguments):
    cache = model.setdefault("_casadi_mpc_cache", {})
    problem = cache.get(cache_key)
    if problem is None:
        problem = _build_casadi_mpc_problem(model=model, cache_key=cache_key, **builder_arguments)
        cache[cache_key] = problem
    return problem


def _casadi_solution_fields_are_finite(solution):
    required_fields = ("x", "f", "g", "lam_x", "lam_g")
    return all(
        field in solution
        and np.all(np.isfinite(np.asarray(solution[field], dtype=float)))
        for field in required_fields
    )


def solve_warmstart_mpc(
    model,
    history_original,
    reference,
    previous_u,
    d_hat=None,
    warm_start_v_seq=None,
    horizon=8,
    move_horizon=None,
    dt=0.01,
    q_state=np.array([30.0, 30.0]),
    r_control=0.01,
    r_delta=0.1,
    terminal_weight=5.0,
    u_min=np.array([-10.0, -10.0]),
    u_max=np.array([10.0, 10.0]),
    du_max=np.array([0.5, 0.5]),
    maxiter=60,
    ftol=1e-4,
    failure_cost=1e12,
    input_matrix=None,
    state_lower_bound=1e-8,
):
    if not (
        _is_bli_model(model)
        or model.get("method") in (None, "raw", "raw_delay")
    ):
        raise ValueError("CasADi MPC supports only Raw-NGRC and BLI-NGRC models.")
    reference = np.asarray(reference, dtype=float)
    previous_u = np.asarray(previous_u, dtype=float)
    q_state = np.asarray(q_state, dtype=float)
    u_min = np.asarray(u_min, dtype=float)
    u_max = np.asarray(u_max, dtype=float)
    du_max = np.asarray(du_max, dtype=float)
    state_dim = len(reference)

    if input_matrix is None:
        input_matrix = np.eye(state_dim)
    input_matrix = np.asarray(input_matrix, dtype=float)
    if (
        input_matrix.ndim != 2
        or input_matrix.shape[0] != state_dim
        or input_matrix.shape[1] < 1
        or not np.all(np.isfinite(input_matrix))
    ):
        raise ValueError("input_matrix must be a finite state_dim-by-input_dim matrix.")
    input_dim = input_matrix.shape[1]
    if state_lower_bound is not None:
        lower = np.asarray(state_lower_bound, dtype=float)
        if lower.ndim == 0:
            lower = np.full(state_dim, float(lower))
        if lower.shape != (state_dim,) or not np.all(np.isfinite(lower)):
            raise ValueError("state_lower_bound must be scalar, a state vector, or None.")
        state_lower_bound = lower

    if d_hat is None:
        d_hat = np.zeros(input_dim)
    d_hat = np.asarray(d_hat, dtype=float)
    state_vectors = (q_state,)
    input_vectors = (previous_u, u_min, u_max, du_max, d_hat)
    if any(vector.shape != (state_dim,) for vector in state_vectors):
        raise ValueError("MPC state weights must match the reference dimension.")
    if any(vector.shape != (input_dim,) for vector in input_vectors):
        raise ValueError("MPC input vectors must match input_matrix columns.")
    if horizon <= 0 or dt <= 0 or maxiter <= 0 or ftol <= 0:
        raise ValueError("MPC horizon, sampling interval, and solver settings must be positive.")
    if move_horizon is None:
        move_horizon = horizon
    if (
        not isinstance(move_horizon, (int, np.integer))
        or move_horizon <= 0
        or move_horizon > horizon
    ):
        raise ValueError("move_horizon must be an integer inside [1, horizon].")
    move_horizon = int(move_horizon)
    if np.any(u_min >= u_max) or np.any(du_max <= 0):
        raise ValueError("MPC input bounds or slew-rate limits are invalid.")

    v_min = u_min + d_hat
    v_max = u_max + d_hat

    if warm_start_v_seq is None:
        initial_full = np.tile(previous_u + d_hat, (horizon, 1))
    else:
        warm = np.asarray(warm_start_v_seq, dtype=float)
        if warm.shape == (horizon, input_dim):
            initial_full = np.vstack([warm[1:], warm[-1]])
        elif warm.shape == (move_horizon, input_dim):
            shifted = np.vstack([warm[1:], warm[-1]])
            initial_full = np.vstack(
                [shifted, np.repeat(shifted[-1:], horizon - move_horizon, axis=0)]
            )
        else:
            initial_full = np.tile(previous_u + d_hat, (horizon, 1))
    initial_actual = project_control_sequence(
        initial_full - d_hat, previous_u, u_min, u_max, du_max
    )
    initial = (initial_actual + d_hat)[:move_horizon]

    def expand_moves(moves):
        moves = np.asarray(moves, dtype=float).reshape(move_horizon, input_dim)
        if move_horizon == horizon:
            return moves
        return np.vstack([moves, np.repeat(moves[-1:], horizon - move_horizon, axis=0)])

    history = np.asarray(history_original, dtype=float)
    required = minimum_history_length(model)
    if history.ndim != 2 or history.shape[1] != state_dim or len(history) < required:
        raise ValueError("Insufficient MPC history or incompatible state dimension.")
    history = history[-required:]
    cache_key = _mpc_cache_key(
        model=model,
        horizon=horizon,
        move_horizon=move_horizon,
        dt=dt,
        q_state=q_state,
        r_control=r_control,
        r_delta=r_delta,
        terminal_weight=terminal_weight,
        du_max=du_max,
        maxiter=maxiter,
        ftol=ftol,
        failure_cost=failure_cost,
        input_matrix=input_matrix,
        state_lower_bound=state_lower_bound,
    )
    problem = _get_casadi_mpc_problem(
        model=model,
        cache_key=cache_key,
        horizon=horizon,
        move_horizon=move_horizon,
        dt=dt,
        q_state=q_state,
        r_control=r_control,
        r_delta=r_delta,
        terminal_weight=terminal_weight,
        du_max=du_max,
        maxiter=maxiter,
        ftol=ftol,
        failure_cost=failure_cost,
        input_matrix=input_matrix,
        state_lower_bound=state_lower_bound,
    )
    if warm_start_v_seq is None:
        problem["lam_x"] = None
        problem["lam_g"] = None
    parameters = np.concatenate([history.ravel(), reference, previous_u, d_hat])
    solve_arguments = {
        "x0": initial.ravel(),
        "p": parameters,
        "lbx": np.tile(v_min, move_horizon),
        "ubx": np.tile(v_max, move_horizon),
        "lbg": problem["lbg"],
        "ubg": problem["ubg"],
    }
    if problem["lam_x"] is not None and problem["lam_g"] is not None:
        solve_arguments["lam_x0"] = problem["lam_x"]
        solve_arguments["lam_g0"] = problem["lam_g"]

    started = time.perf_counter()
    solution = None
    error_message = ""
    try:
        solution = problem["solver"](**solve_arguments)
        solver_stats = problem["solver"].stats()
    except RuntimeError as error:
        solver_stats = problem["solver"].stats()
        error_message = str(error)
    elapsed = time.perf_counter() - started
    candidate = None
    if solution is not None:
        candidate = np.asarray(solution['x'], dtype=float).reshape(move_horizon, input_dim)
    solution_fields_finite = (
        solution is not None and _casadi_solution_fields_are_finite(solution)
    )
    constraint_tolerance = max(1e-7, 10.0 * float(ftol))
    feasible = False
    valid_prediction = False
    objective_below_failure = False
    if solution_fields_finite and candidate is not None:
        flat_candidate = candidate.ravel()
        constraint_values = np.asarray(solution["g"], dtype=float).reshape(-1)
        feasible = bool(
            np.all(flat_candidate >= np.tile(v_min, move_horizon) - constraint_tolerance)
            and np.all(flat_candidate <= np.tile(v_max, move_horizon) + constraint_tolerance)
            and np.all(constraint_values >= problem["lbg"] - constraint_tolerance)
            and np.all(constraint_values <= problem["ubg"] + constraint_tolerance)
        )
        candidate_sequence = expand_moves(candidate)
        candidate_prediction = rollout_ngrc_controlled(
            model=model,
            history_original=history,
            controls=candidate_sequence @ input_matrix.T,
            dt=dt,
            state_lower_bound=state_lower_bound,
        )
        valid_prediction = bool(
            candidate_prediction.shape == (horizon + 1, state_dim)
            and np.all(np.isfinite(candidate_prediction))
        )
        objective_value = float(np.asarray(solution["f"]).reshape(-1)[0])
        objective_below_failure = objective_value < float(failure_cost)
    success = bool(
        solver_stats.get("success", False)
        and solution_fields_finite
        and feasible
        and valid_prediction
        and objective_below_failure
    )
    if success:
        problem["lam_x"] = np.asarray(solution["lam_x"], dtype=float).reshape(-1)
        problem["lam_g"] = np.asarray(solution["lam_g"], dtype=float).reshape(-1)
        sequence = expand_moves(candidate)
    else:
        problem["lam_x"] = None
        problem["lam_g"] = None
        sequence = expand_moves(initial)
    message = str(solver_stats.get("return_status", error_message))
    result = SimpleNamespace(
        success=bool(success),
        status=0 if success else 1,
        message=message,
        x=(candidate if success else initial).ravel(),
        fun=(
            float(np.asarray(solution['f']).reshape(-1)[0]) if success else float(failure_cost)
        ),
        nit=int(solver_stats.get("iter_count", 0)),
        nfev=int(solver_stats.get("n_call_nlp_f", 0)),
        solve_time_seconds=float(elapsed),
        solver="IPOPT",
    )
    actual_sequence = project_control_sequence(sequence - d_hat, previous_u, u_min, u_max, du_max)
    virtual_sequence = actual_sequence + d_hat
    return (actual_sequence[0].copy(), virtual_sequence[0].copy(), result, virtual_sequence)


# 7. Closed-loop simulation and fair comparison


def _control_dt(results):
    config = results.get("config")
    if config is None or not hasattr(config, "identification_dt"):
        raise ValueError("Model results do not expose identification_dt.")
    return float(config.identification_dt)

def _smooth_blili_feedback_history(history, window=9, degree=2):
    """Causally smooth the feedback history so BLI's continuous-time rate
    field sees inputs from the same (denoised) distribution it was trained on.

    Only used for BLI models without an ESO, where the raw measurements are
    pushed directly into the MPC planning history.
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or len(history) < 3:
        return history
    effective_window = min(int(window), len(history))
    if effective_window < 3:
        return history
    effective_degree = min(int(degree), effective_window - 1)
    return ROBUST_NGRC.causal_savgol_filter(
        history, window=effective_window, degree=effective_degree
    )


def simulate_controller_from_handoff(
    model,
    config,
    controller_type,
    handoff_time,
    handoff_true_state,
    handoff_measurement,
    model_history,
    true_history,
    future_measurement_noise=None,
    plant_step=None,
):
    if model.get("method") not in ("raw", "raw_delay", "bli", "bli_delay"):
        raise ValueError("Unsupported control model method.")
    dt = float(model.get('identification_dt', config.internal_dt * config.control_stride))
    config.validate(dt)
    if controller_type not in ("mpc", "eso_adr_mpc"):
        raise ValueError("controller_type must be 'mpc' or 'eso_adr_mpc'.")

    reference = np.asarray(config.reference, dtype=float)
    dim = len(reference)
    input_matrix = config.resolved_input_matrix()
    input_dimension = input_matrix.shape[1]
    input_pseudoinverse = np.linalg.pinv(input_matrix)
    state_lower_bound = config.resolved_state_lower_bound()
    handoff_true_state = np.asarray(handoff_true_state, dtype=float)
    handoff_measurement = np.asarray(handoff_measurement, dtype=float)
    if handoff_true_state.shape != (dim,) or handoff_measurement.shape != (dim,):
        raise ValueError("Handoff state and measurement dimensions must match reference.")
    if not np.isfinite(handoff_time):
        raise ValueError("handoff_time must be finite.")
    if np.asarray(model["W_out"]).shape[0] != dim:
        raise ValueError("Model output dimension and control dimension do not match.")

    required = minimum_history_length(model)
    model_history = np.asarray(model_history, dtype=float)
    true_history = np.asarray(true_history, dtype=float)
    if (
        model_history.ndim != 2
        or true_history.ndim != 2
        or model_history.shape[1] != dim
        or true_history.shape[1] != dim
        or len(model_history) < required
        or len(true_history) < required
    ):
        raise ValueError("Handoff histories are too short or have the wrong dimension.")
    feedback_history = model_history[-required:].copy()
    initial_measurement = handoff_measurement
    feedback_history[-1] = initial_measurement
    rolling_true_history = true_history[-required:].copy()
    rolling_true_history[-1] = handoff_true_state

    num_steps = int(round(config.control_duration / dt))
    if future_measurement_noise is None:
        rng = np.random.default_rng(config.random_seed)
        future_measurement_noise = config.resolved_noise_std() * rng.standard_normal(
            (num_steps, dim)
        )
    future_measurement_noise = np.asarray(future_measurement_noise, dtype=float)
    if future_measurement_noise.shape != (num_steps, dim):
        raise ValueError("future_measurement_noise has the wrong shape.")
    if not np.all(np.isfinite(future_measurement_noise)):
        raise ValueError("future_measurement_noise must be finite.")

    times = float(handoff_time) + np.arange(num_steps + 1) * dt
    X_true = np.zeros((num_steps + 1, dim))
    X_meas = np.zeros_like(X_true)
    X_feedback = np.zeros_like(X_true)
    U_actual = np.zeros((num_steps, input_dimension))
    V_virtual = np.zeros((num_steps, input_dimension))
    D_hat = np.zeros((num_steps + 1, input_dimension))
    D_raw = np.zeros_like(D_hat)
    D_true = np.zeros((num_steps, input_dimension))
    optimizer_success = np.zeros(num_steps, dtype=bool)
    optimizer_solve_time = np.zeros(num_steps, dtype=float)
    optimizer_iterations = np.zeros(num_steps, dtype=int)
    optimizer_nfev = np.zeros(num_steps, dtype=int)
    X_true[0] = handoff_true_state
    X_meas[0] = handoff_measurement

    observer = None
    if controller_type == "eso_adr_mpc":
        observer = NGRCResidualESO(
            dim=dim,
            observer_bandwidth=config.observer_bandwidth,
            input_matrix=input_matrix,
            state_lower_bound=state_lower_bound,
        )
        observer.reset(initial_measurement)
    X_feedback[0] = initial_measurement

    previous_u = np.zeros(input_dimension)
    if plant_step is None:
        plant_step = euler_step_controlled_plant_substeps
    warm_start_virtual = None
    for step in range(num_steps):
        if step > 0:
            feedback_measurement = X_meas[step]
            if observer is not None:
                feedback_measurement, state_disturbance = observer.update(
                    measurement=X_meas[step],
                    previous_u=previous_u,
                    dt=dt,
                    model=model,
                    observer_history=feedback_history,
                )
                D_hat[step] = input_pseudoinverse @ state_disturbance
                D_raw[step] = D_hat[step]
            X_feedback[step] = feedback_measurement
            feedback_history = append_and_trim(feedback_history, X_feedback[step], required)
            # BLI's W_out was trained on smoothed latent states, whereas the
            # MPC planning history feeds it raw measurements. When no ESO is
            # filtering the feedback, the resulting input distribution is
            # wider than the training envelope, which distorts the rate field
            # and shows up as a larger steady-state error. A short causal SG
            # filter restores the training/deployment agreement.
            if observer is None and model.get("method") in ("bli", "bli_delay"):
                smoothed = _smooth_blili_feedback_history(feedback_history)
                feedback_history[-1] = smoothed[-1]

        disturbance_hat = (
            D_hat[step] if observer is not None else np.zeros(input_dimension)
        )
        control, virtual_control, optimization, warm_start_virtual = solve_warmstart_mpc(
            model=model,
            history_original=feedback_history,
            reference=reference,
            previous_u=previous_u,
            d_hat=disturbance_hat,
            warm_start_v_seq=warm_start_virtual,
            horizon=config.horizon,
            move_horizon=config.move_horizon,
            dt=dt,
            q_state=config.q_state,
            r_control=config.r_control,
            r_delta=config.r_delta,
            terminal_weight=config.terminal_weight,
            u_min=config.u_min,
            u_max=config.u_max,
            du_max=config.du_max,
            maxiter=config.mpc_maxiter,
            ftol=config.mpc_ftol,
            failure_cost=config.prediction_failure_cost,
            input_matrix=input_matrix,
            state_lower_bound=state_lower_bound,
        )
        U_actual[step] = control
        V_virtual[step] = virtual_control
        optimizer_success[step] = bool(optimization.success)
        optimizer_solve_time[step] = float(optimization.solve_time_seconds)
        optimizer_iterations[step] = int(optimization.nit)
        optimizer_nfev[step] = int(optimization.nfev)

        nominal_true_next = ngrc_one_step_nominal(model, rolling_true_history)
        X_true[step + 1] = plant_step(
            X_true[step],
            times[step],
            control,
            internal_dt=config.internal_dt,
            substeps=config.control_stride,
            with_disturbance=config.with_disturbance,
        )
        state_residual = (
            (X_true[step + 1] - nominal_true_next) / dt
            - input_matrix @ control
        )
        D_true[step] = input_pseudoinverse @ state_residual
        rolling_true_history = append_and_trim(rolling_true_history, X_true[step + 1], required)
        X_meas[step + 1] = X_true[step + 1] + future_measurement_noise[step]
        previous_u = control.copy()

    final_measurement = X_meas[-1]
    if observer is not None:
        final_measurement, state_disturbance = observer.update(
            measurement=X_meas[-1],
            previous_u=previous_u,
            dt=dt,
            model=model,
            observer_history=feedback_history,
        )
        D_hat[-1] = input_pseudoinverse @ state_disturbance
        D_raw[-1] = D_hat[-1]
    X_feedback[-1] = final_measurement
    if observer is not None:
        update_count = observer.update_count
        magnitude_saturations = 0
        slew_saturations = 0
    else:
        update_count = 0
        magnitude_saturations = 0
        slew_saturations = 0

    return {
        "controller_type": controller_type,
        "model_key": model.get("method", config.model_key),
        "selected_alpha": float(model.get("selected_alpha", np.nan)),
        "t": times,
        "X_true": X_true,
        "X_meas": X_meas,
        "X_feedback": X_feedback,
        "U_actual": U_actual,
        "V_virtual": V_virtual,
        "D_hat": D_hat,
        "D_raw": D_raw,
        "D_true": D_true,
        "x_ref": reference,
        "dt_control": dt,
        "internal_dt": config.internal_dt,
        "control_stride": config.control_stride,
        "optimizer_success": optimizer_success,
        "optimizer_name": "IPOPT",
        "optimizer_solve_time": optimizer_solve_time,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_nfev": optimizer_nfev,
        "observer_update_count": update_count,
        "observer_magnitude_saturations": magnitude_saturations,
        "observer_slew_saturations": slew_saturations,
        "horizon": config.horizon,
        "move_horizon": config.move_horizon,
    }


def simulate_controller(results, config, controller_type, measurement_noise=None, plant_step=None):
    dt = _control_dt(results)
    config.validate(dt)
    if config.model_key not in results:
        raise KeyError(f"Model result '{config.model_key}' is unavailable.")
    model = results[config.model_key]
    dim = len(np.asarray(config.reference))
    train_len = int(model.get("train_len", 0))
    if train_len <= 0:
        train_len = min(len(results["t"]) - 1, len(model["X_input"]) - 1)
    if train_len >= len(results["X_clean"]):
        raise ValueError("Training boundary lies outside the clean-state record.")

    num_steps = int(round(config.control_duration / dt))
    if measurement_noise is None:
        rng = np.random.default_rng(config.random_seed)
        measurement_noise = config.resolved_noise_std() * rng.standard_normal((num_steps + 1, dim))
    measurement_noise = np.asarray(measurement_noise, dtype=float)
    if measurement_noise.shape != (num_steps + 1, dim):
        raise ValueError("measurement_noise has the wrong shape.")

    required = minimum_history_length(model)
    start = train_len - required + 1
    if start < 0:
        raise ValueError("Not enough pre-control samples to initialize NGRC history.")
    handoff_true_state = np.asarray(results["X_clean"][train_len], dtype=float)
    handoff_measurement = handoff_true_state + measurement_noise[0]
    return simulate_controller_from_handoff(
        model=model,
        config=config,
        controller_type=controller_type,
        handoff_time=float(results["t"][train_len]),
        handoff_true_state=handoff_true_state,
        handoff_measurement=handoff_measurement,
        model_history=np.asarray(model['X_input'][start:train_len + 1], dtype=float),
        true_history=np.asarray(results['X_clean'][start:train_len + 1], dtype=float),
        future_measurement_noise=measurement_noise[1:],
        plant_step=plant_step,
    )


def run_control_comparison(results, config=None, plant_step=None):
    if config is None:
        config = ControlConfig()
    dt = _control_dt(results)
    config.validate(dt)
    dim = len(np.asarray(config.reference))
    num_steps = int(round(config.control_duration / dt))
    rng = np.random.default_rng(config.random_seed)
    shared_noise = config.resolved_noise_std() * rng.standard_normal((num_steps + 1, dim))

    ordinary = simulate_controller(
        results, config, "mpc", measurement_noise=shared_noise, plant_step=plant_step
    )
    eso = simulate_controller(
        results,
        config,
        "eso_adr_mpc",
        measurement_noise=shared_noise,
        plant_step=plant_step,
    )
    return ordinary, eso


# 8. Metrics, reporting, and plots


def tracking_rmse(simulation):
    error = np.asarray(simulation["X_true"], dtype=float) - _control_reference(simulation)
    return float(np.sqrt(np.mean(error**2)))


def _control_reference(simulation):
    key = "x_ref" if "x_ref" in simulation else "reference"
    reference = np.asarray(simulation[key], dtype=float)
    if reference.ndim != 1 or not np.all(np.isfinite(reference)):
        raise ValueError("Control reference must be a finite state vector.")
    return reference


def _control_array(simulation):
    key = "U_actual" if "U_actual" in simulation else "U"
    controls = np.asarray(simulation[key], dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    if controls.ndim != 2 or not np.all(np.isfinite(controls)):
        raise ValueError("Control input must be a finite one- or two-dimensional array.")
    return controls


def final_window_mask(simulation, window_duration=2.0):
    time = np.asarray(simulation["t"], dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("Control time must be a finite strictly increasing vector.")
    if not np.isfinite(window_duration) or window_duration <= 0.0:
        raise ValueError("Final-window duration must be finite and positive.")
    duration = min(float(window_duration), float(time[-1] - time[0]))
    mask = time >= time[-1] - duration - 1e-12
    return mask, duration


def steady_state_rmse(simulation, window_duration=2.0):
    mask, _ = final_window_mask(simulation, window_duration)
    error = np.asarray(simulation["X_true"], dtype=float)[mask] - _control_reference(simulation)
    return float(np.sqrt(np.mean(error**2)))


def final_tracking_error(simulation):
    return float(
        np.linalg.norm(
            np.asarray(simulation["X_true"], dtype=float)[-1]
            - _control_reference(simulation)
        )
    )


def control_energy(simulation):
    controls = _control_array(simulation)
    dt = float(
        simulation.get(
            "dt_control",
            np.asarray(simulation["t"], dtype=float)[1]
            - np.asarray(simulation["t"], dtype=float)[0],
        )
    )
    return float(np.sum(controls**2) * dt)


def control_increment_rms(simulation):
    controls = _control_array(simulation)
    if len(controls) < 2:
        return 0.0
    increments = np.diff(controls, axis=0)
    return float(np.sqrt(np.mean(increments**2)))


def disturbance_estimation_rmse(simulation):
    """Paper E_d: input-space residual error over applied control intervals."""
    if "D_hat" not in simulation or "D_true" not in simulation:
        return None
    actual = np.asarray(simulation["D_true"], dtype=float)
    estimated = np.asarray(simulation["D_hat"], dtype=float)
    controls = _control_array(simulation)
    if actual.shape != controls.shape or estimated.shape != (len(controls) + 1, controls.shape[1]):
        raise ValueError("Disturbance arrays must align with applied inputs and recorded states.")
    # The final observer update has no following applied input interval.
    return float(np.sqrt(np.mean((estimated[:-1] - actual) ** 2)))


def control_metric_panel(simulation, window_duration=2.0):
    time = np.asarray(simulation["t"], dtype=float)
    states = np.asarray(simulation["X_true"], dtype=float)
    reference = _control_reference(simulation)
    if states.shape != (len(time), len(reference)) or not np.all(np.isfinite(states)):
        raise ValueError("Control states must match time and reference dimensions.")
    mask, final_duration = final_window_mask(simulation, window_duration)
    error = states - reference
    final_error = error[mask]
    dt = float(simulation.get('dt_control', time[1] - time[0]))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Control sampling interval must be finite and positive.")
    controls = _control_array(simulation)
    increments = np.diff(controls, axis=0)
    elapsed = time - time[0]
    error_norm = np.linalg.norm(error, axis=1)
    reference_norm = float(np.linalg.norm(reference))
    settling_tolerance = 0.02 * reference_norm if reference_norm > 0.0 else 0.02
    settling_index = next(
        (
            index
            for index in range(len(time))
            if np.all(error_norm[index:] <= settling_tolerance + 1e-12)
        ),
        None,
    )
    settling_time = (
        None if settling_index is None else float(elapsed[settling_index])
    )
    return {
        "tracking_rmse": float(np.sqrt(np.mean(error**2))),
        "per_state_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "final_window_duration": float(final_duration),
        "final_window_rmse": float(np.sqrt(np.mean(final_error**2))),
        "final_window_per_state_rmse": np.sqrt(np.mean(final_error ** 2, axis=0)).tolist(),
        "final_window_per_state_mae": np.mean(np.abs(final_error), axis=0).tolist(),
        "final_tracking_error": float(error_norm[-1]),
        "final_abs_error": np.abs(error[-1]).tolist(),
        "settling_time_2pct": settling_time,
        "iae": float(dt * np.sum(np.abs(error[:-1]))),
        "itae": float(dt * np.sum(elapsed[:-1, None] * np.abs(error[:-1]))),
        "control_energy": float(dt * np.sum(controls**2)),
        "control_increment_rms": (
            float(np.sqrt(np.mean(increments**2))) if len(increments) else 0.0
        ),
        "control_total_variation": float(np.sum(np.abs(increments))),
        "disturbance_estimation_rmse": disturbance_estimation_rmse(simulation),
        "peak_abs_control": (
            float(np.max(np.abs(controls))) if controls.size else 0.0
        ),
    }


def print_control_summary(
    ordinary,
    eso,
    final_window_duration=2.0,
    raw_mpc=None,
    raw_delay_mpc=None,
    raw_adr_mpc=None,
):
    print("========== BLI-NGRC closed-loop comparison ==========")
    print(
        f"Model: {ordinary['model_key']} | normalized alpha = "
        f"{ordinary['selected_alpha']:.8g} | dt = {ordinary['dt_control']:.4f}"
    )
    if raw_mpc is None and raw_delay_mpc is None and raw_adr_mpc is None:
        simulations = (('BLI-NGRC+MPC', ordinary), ('BLI-NGRC+ESO-ADR+MPC', eso))
    else:
        simulations = []
        if raw_mpc is not None:
            simulations.append(("Raw-NGRC+MPC", raw_mpc))
        if raw_delay_mpc is not None:
            simulations.append(("Raw-delay-NGRC+MPC", raw_delay_mpc))
        simulations.extend((('BLI-NGRC+MPC', ordinary), ('BLI-NGRC+ESO-ADR+MPC', eso)))
        if raw_adr_mpc is not None:
            simulations.append(("Raw-NGRC+ESO-ADR+MPC", raw_adr_mpc))
    for label, simulation in simulations:
        panel = control_metric_panel(simulation, final_window_duration)
        disturbance_rmse = panel["disturbance_estimation_rmse"]
        disturbance_text = "N/A" if disturbance_rmse is None else f"{disturbance_rmse:.6f}"
        print(
            f"{label:28s} | tracking RMSE = {tracking_rmse(simulation):.6f} | "
            f"final-window RMSE = {steady_state_rmse(simulation, final_window_duration):.6f} | "
            f"final error = {final_tracking_error(simulation):.6f} | "
            f"control energy = {control_energy(simulation):.6f} | "
            f"control increment RMS = {control_increment_rms(simulation):.6f} | "
            f"optimizer success = {100.0 * np.mean(simulation['optimizer_success']):.2f}% | "
            f"{simulation['optimizer_name']} mean solve time = "
            f"{1e3 * np.mean(simulation['optimizer_solve_time']):.3f} ms | "
            f"total solve time = {np.sum(simulation['optimizer_solve_time']):.3f} s | "
            f"mean iterations = {np.mean(simulation['optimizer_iterations']):.2f}"
        )
        print(
            f"  IAE = {panel['iae']:.6f} | ITAE = {panel['itae']:.6f} | "
            f"control TV = {panel['control_total_variation']:.6f} | "
            f"equivalent disturbance RMSE = {disturbance_text} | "
            f"final-window duration = {panel['final_window_duration']:.3f}"
        )
    print("=========================================================")


def plot_control_comparison(
    ordinary,
    eso,
    raw_mpc=None,
    raw_delay_mpc=None,
    raw_adr_mpc=None,
    output_path=None,
    show=True,
):
    t = ordinary["t"]
    t_u = t[:-1]
    reference = ordinary["x_ref"]
    state_names = ("x", "y")
    control_names = ("u_x", "u_y")
    figure, axes = plt.subplots(4, 2, figsize=(13, 13))

    for channel in range(2):
        axis = axes[0, channel]
        if raw_mpc is not None:
            axis.plot(t, raw_mpc["X_true"][:, channel], label="Raw-NGRC+MPC")
            ordinary_label = "BLI-NGRC+MPC"
            eso_label = "BLI-NGRC+ESO-ADR+MPC"
        else:
            ordinary_label = "BLI-NGRC+MPC"
            eso_label = "BLI-NGRC+ESO-ADR+MPC"
        if raw_delay_mpc is not None:
            axis.plot(t, raw_delay_mpc['X_true'][:, channel], label='Raw-delay-NGRC+MPC')
        if raw_adr_mpc is not None:
            axis.plot(t, raw_adr_mpc['X_true'][:, channel], label='Raw-NGRC+ESO-ADR+MPC')
        axis.plot(t, ordinary["X_true"][:, channel], label=ordinary_label)
        axis.plot(t, eso["X_true"][:, channel], label=eso_label)
        axis.axhline(reference[channel], color="tab:blue", linestyle="--", label="Target")
        axis.set_title(f"Closed-loop state: {state_names[channel]}")
        axis.set_xlabel("Time")
        axis.set_ylabel(state_names[channel])
        axis.grid(alpha=0.3)
        axis.legend()

        axis = axes[1, channel]
        if raw_mpc is not None:
            axis.plot(t_u, raw_mpc['U_actual'][:, channel], label='Raw-NGRC+MPC')
        if raw_delay_mpc is not None:
            axis.plot(t_u, raw_delay_mpc['U_actual'][:, channel], label='Raw-delay-NGRC+MPC')
        if raw_adr_mpc is not None:
            axis.plot(t_u, raw_adr_mpc['U_actual'][:, channel], label='Raw-NGRC+ESO-ADR+MPC')
        axis.plot(t_u, ordinary["U_actual"][:, channel], label=ordinary_label)
        axis.plot(t_u, eso["U_actual"][:, channel], label=eso_label)
        axis.set_title(f"Actual control input: {control_names[channel]}")
        axis.set_xlabel("Time")
        axis.set_ylabel(control_names[channel])
        axis.grid(alpha=0.3)
        axis.legend()

        axis = axes[2, channel]
        estimation_error = eso["X_feedback"][:, channel] - eso["X_true"][:, channel]
        axis.plot(t, estimation_error, color="tab:green")
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(f"ESO state estimation error: {state_names[channel]}")
        axis.set_xlabel("Time")
        axis.set_ylabel("estimate - true")
        axis.grid(alpha=0.3)

        axis = axes[3, channel]
        axis.plot(t_u, eso["D_true"][:, channel], label="Equivalent disturbance")
        axis.plot(t, eso["D_hat"][:, channel], "--", label="ESO estimate")
        axis.set_title(f"Equivalent disturbance: channel {state_names[channel]}")
        axis.set_xlabel("Time")
        axis.set_ylabel("disturbance")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def run_full_experiment(model_config=None, control_config=None, make_plots=True, verbose=True):
    if model_config is None:
        model_config = ROBUST_NGRC.ExperimentConfig()
    if control_config is None:
        control_config = ControlConfig()

    results = ROBUST_NGRC.run_experiment(
        config=model_config,
        methods=(control_config.model_key,),
        make_plots=False,
        verbose=verbose,
    )
    ordinary, eso = run_control_comparison(results, control_config)
    if verbose:
        print_control_summary(
            ordinary,
            eso,
            final_window_duration=control_config.steady_state_window,
        )
    if make_plots:
        plot_control_comparison(ordinary, eso)
    return results, ordinary, eso


if __name__ == "__main__":
    run_full_experiment()
