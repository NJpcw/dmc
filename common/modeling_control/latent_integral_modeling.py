"""Bi-level latent integral-consistent next-generation reservoir computing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve


@dataclass
class BLIConfig:
    training_samples: int = 300
    dt: float = 0.01
    horizons: tuple = (1, 2, 5, 10, 20)
    alpha_grid: tuple = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    alternating_iterations: int = 3
    whittaker_penalty: float = 100.0
    latent_smoothness_weight: float = 0.2
    latent_dynamics_weight: float = 1.0
    endpoint_weight: float = 4.0
    latent_bound_margin_std: float = 6.0
    latent_max_nfev: int = 12
    objective_tolerance: float = 1e-4
    extrapolation_margin: float = 2.0
    validation_fit_fractions: tuple = (0.55, 0.70, 0.85)
    validation_horizon: int = 20
    validation_anchors_per_fold: int = 12
    divergence_penalty: float = 1e3
    feature_k: int = 1
    feature_s: int = 1

    def validate(self):
        if not isinstance(self.training_samples, (int, np.integer)) or self.training_samples < 20:
            raise ValueError("training_samples must be an integer of at least 20.")
        positive_scalars = (
            self.dt,
            self.whittaker_penalty,
            self.latent_smoothness_weight,
            self.latent_dynamics_weight,
            self.endpoint_weight,
            self.latent_bound_margin_std,
            self.objective_tolerance,
            self.extrapolation_margin,
            self.divergence_penalty,
        )
        if not np.all(np.isfinite(positive_scalars)) or any(value <= 0 for value in positive_scalars):
            raise ValueError("Every positive BLI scalar must be finite and positive.")
        if not isinstance(self.alternating_iterations, (int, np.integer)) or self.alternating_iterations <= 0:
            raise ValueError("alternating_iterations must be a positive integer.")
        if not isinstance(self.latent_max_nfev, (int, np.integer)) or self.latent_max_nfev <= 0:
            raise ValueError("latent_max_nfev must be a positive integer.")
        if not isinstance(self.feature_k, (int, np.integer)) or self.feature_k <= 0:
            raise ValueError("feature_k must be a positive integer.")
        if not isinstance(self.feature_s, (int, np.integer)) or self.feature_s <= 0:
            raise ValueError("feature_s must be a positive integer.")
        if not self.horizons or max(self.horizons) >= self.training_samples:
            raise ValueError("BLI horizons must fit inside the training record.")
        fractions = np.asarray(self.validation_fit_fractions, dtype=float)
        if (
            fractions.ndim != 1
            or len(fractions) == 0
            or np.any(fractions <= 0)
            or np.any(fractions >= 1)
            or np.any(np.diff(fractions) <= 0)
        ):
            raise ValueError("validation_fit_fractions must increase inside (0, 1).")
        if not isinstance(self.validation_horizon, (int, np.integer)) or self.validation_horizon <= 0:
            raise ValueError("validation_horizon must be a positive integer.")
        if (
            not isinstance(self.validation_anchors_per_fold, (int, np.integer))
            or self.validation_anchors_per_fold <= 0
        ):
            raise ValueError("validation_anchors_per_fold must be a positive integer.")
        alphas = np.asarray(self.alpha_grid, dtype=float)
        if alphas.ndim != 1 or len(alphas) == 0 or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
            raise ValueError("alpha_grid must contain positive finite values.")


def estimate_observation_noise_std(observations, minimum_std=1e-8):
    """Estimate one observation-noise scale per state via second-difference MAD."""
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError('observations must be a finite 2-D array with at least three rows.')
    if not np.isfinite(minimum_std) or minimum_std <= 0.0:
        raise ValueError("minimum_std must be finite and positive.")
    second_differences = np.diff(values, n=2, axis=0)
    centered = second_differences - np.median(second_differences, axis=0)
    estimate = (
        1.4826
        / np.sqrt(6.0)
        * np.median(np.abs(centered), axis=0)
    )
    return np.maximum(float(minimum_std), estimate)


class RobustStateScaler:
    """Channel-wise robust centering and scaling for BLI latent states."""

    def __init__(self, scale_floor=1e-8):
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive.")
        self.scale_floor = float(scale_floor)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a non-empty finite two-dimensional array.")
        self.center_ = np.median(X, axis=0)
        lower, upper = np.percentile(X, [25.0, 75.0], axis=0)
        robust_scale = (upper - lower) / 1.349
        standard_scale = np.std(X, axis=0)
        fallback = np.maximum(standard_scale, self.scale_floor)
        self.scale_ = np.where(robust_scale > self.scale_floor, robust_scale, fallback)
        self.scale_ = np.maximum(self.scale_, self.scale_floor)
        return self

    def transform(self, X):
        if not hasattr(self, "center_"):
            raise RuntimeError("RobustStateScaler must be fitted before use.")
        X = np.asarray(X, dtype=float)
        return (X - self.center_) / self.scale_

    def inverse_transform(self, Z):
        if not hasattr(self, "center_"):
            raise RuntimeError("RobustStateScaler must be fitted before use.")
        Z = np.asarray(Z, dtype=float)
        return Z * self.scale_ + self.center_


class QuadraticFeatureTransformer:
    """Guarded current-state total-degree-two monomial library."""

    def __init__(self, extrapolation_margin=2.0):
        if not np.isfinite(extrapolation_margin) or extrapolation_margin <= 0:
            raise ValueError("extrapolation_margin must be finite and positive.")
        self.extrapolation_margin = float(extrapolation_margin)

    def fit(self, states_scaled):
        states_scaled = np.asarray(states_scaled, dtype=float)
        if (
            states_scaled.ndim != 2
            or states_scaled.shape[1] < 1
            or len(states_scaled) == 0
            or not np.all(np.isfinite(states_scaled))
        ):
            raise ValueError("NGRC features require a non-empty finite [n, d] array.")
        self.state_dim_ = int(states_scaled.shape[1])
        self.training_limit_ = np.maximum(np.max(np.abs(states_scaled), axis=0), 1.0)
        self.pairs_ = tuple(
            (left, right)
            for left in range(self.state_dim_)
            for right in range(left, self.state_dim_)
        )
        self.num_features_ = 1 + self.state_dim_ + len(self.pairs_)
        return self

    def signature(self):
        if not hasattr(self, "state_dim_"):
            raise RuntimeError("QuadraticFeatureTransformer must be fitted before use.")
        return {
            "name": "guarded_total_degree_two",
            "state_dim": self.state_dim_,
            "pairs": self.pairs_,
            "num_features": self.num_features_,
            "extrapolation_margin": self.extrapolation_margin,
        }

    def _guard(self, states_scaled):
        if not hasattr(self, "training_limit_"):
            raise RuntimeError("QuadraticFeatureTransformer must be fitted before use.")
        states_scaled = np.asarray(states_scaled, dtype=float)
        if (
            states_scaled.ndim != 2
            or states_scaled.shape[1] != self.state_dim_
            or not np.all(np.isfinite(states_scaled))
        ):
            raise ValueError("states_scaled must have fitted state dimension and be finite.")
        magnitude = np.abs(states_scaled)
        exact_limit = self.training_limit_ * (1.0 + self.extrapolation_margin)
        excess = np.maximum(magnitude - exact_limit, 0.0)
        guarded_magnitude = exact_limit + exact_limit * np.tanh(excess / exact_limit)
        return np.where(
            magnitude <= exact_limit,
            states_scaled,
            np.sign(states_scaled) * guarded_magnitude,
        )

    def transform(self, states_scaled):
        states_scaled = np.asarray(states_scaled, dtype=float)
        guarded = self._guard(states_scaled)
        quadratic = [
            guarded[:, left] * guarded[:, right]
            for left, right in self.pairs_
        ]
        return np.column_stack([np.ones(len(guarded)), guarded, *quadratic])

    def transform_one(self, state_scaled):
        state_scaled = np.asarray(state_scaled, dtype=float)
        if state_scaled.shape != (self.state_dim_,):
            raise ValueError("state_scaled must have the fitted state dimension.")
        guarded = self.guard_one(state_scaled)
        return np.array(
            [1.0, *guarded, *(guarded[left] * guarded[right] for left, right in self.pairs_)],
            dtype=float,
        )

    def guard_one(self, state_scaled):
        """Allocation-light scalar guard used in repeated Euler/MPC evaluations."""
        if not hasattr(self, "training_limit_"):
            raise RuntimeError("QuadraticFeatureTransformer must be fitted before use.")
        state_scaled = np.asarray(state_scaled, dtype=float)
        if state_scaled.shape != (self.state_dim_,) or not np.all(np.isfinite(state_scaled)):
            raise ValueError("state_scaled must have the fitted state dimension and be finite.")
        guarded = np.empty(self.state_dim_, dtype=float)
        for channel in range(self.state_dim_):
            value = float(state_scaled[channel])
            magnitude = abs(value)
            exact_limit = float(self.training_limit_[channel] * (1.0 + self.extrapolation_margin))
            if magnitude <= exact_limit:
                guarded[channel] = value
            else:
                bounded = exact_limit + exact_limit * math.tanh(
                    (magnitude - exact_limit) / exact_limit
                )
                guarded[channel] = math.copysign(bounded, value)
        return guarded


# Historical name retained for inactive callers and old serialized metadata.
ChebyshevFeatureTransformer = QuadraticFeatureTransformer


def build_causal_embedding(states_scaled, k=1, s=1):
    """Return causal state taps ordered from current to oldest."""
    states_scaled = np.asarray(states_scaled, dtype=float)
    if (
        states_scaled.ndim != 2
        or states_scaled.shape[1] < 1
        or len(states_scaled) == 0
        or not np.all(np.isfinite(states_scaled))
    ):
        raise ValueError("BLI embedding states must be a non-empty finite [n, d] array.")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("BLI embedding k must be a positive integer.")
    if not isinstance(s, (int, np.integer)) or s <= 0:
        raise ValueError("BLI embedding s must be a positive integer.")
    first_valid = (int(k) - 1) * int(s)
    if len(states_scaled) <= first_valid:
        raise ValueError("BLI states are too short for the requested embedding.")
    indices = np.arange(first_valid, len(states_scaled), dtype=int)
    embedded = np.asarray(
        [
            np.concatenate([states_scaled[index - lag * int(s)] for lag in range(int(k))])
            for index in indices
        ],
        dtype=float,
    )
    return embedded, indices


def build_integral_design(
    states_scaled,
    transformer,
    dt,
    horizons=(1, 2, 5, 10, 20),
    feature_k=1,
    feature_s=1,
):
    """Build trapezoidal weak-form constraints for a continuous NGRC readout."""
    states_scaled = np.asarray(states_scaled, dtype=float)
    if (
        states_scaled.ndim != 2
        or states_scaled.shape[1] < 1
        or len(states_scaled) < 2
        or not np.all(np.isfinite(states_scaled))
    ):
        raise ValueError("states_scaled must be a finite [n, d] array with n >= 2.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    horizons = tuple(horizons)
    if (
        not horizons
        or any(not isinstance(h, (int, np.integer)) or h <= 0 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must contain distinct positive integers.")

    embedded, valid_indices = build_causal_embedding(states_scaled, k=feature_k, s=feature_s)
    features = transformer.transform(embedded)
    first_valid = int(valid_indices[0])
    design_columns = []
    target_columns = []
    starts = []
    used_horizons = []
    for horizon in sorted(horizons):
        for start in range(first_valid, len(states_scaled) - horizon):
            stop = start + horizon
            feature_start = start - first_valid
            feature_stop = stop - first_valid
            integral = 0.5 * (features[feature_start] + features[feature_stop])
            if horizon > 1:
                integral = integral + np.sum(features[feature_start + 1:feature_stop], axis=0)
            design_columns.append(float(dt) * integral)
            target_columns.append(states_scaled[stop] - states_scaled[start])
            starts.append(start)
            used_horizons.append(horizon)
    if not design_columns:
        raise ValueError("No integral constraints fit inside the state record.")
    return (
        np.asarray(design_columns, dtype=float).T,
        np.asarray(target_columns, dtype=float).T,
        {
            "starts": np.asarray(starts, dtype=int),
            "horizons": np.asarray(used_horizons, dtype=int),
            "feature_k": int(feature_k),
            "feature_s": int(feature_s),
            "first_valid_index": first_valid,
        },
    )


def fit_integral_readout(
    states_scaled,
    transformer,
    dt,
    horizons=(1, 2, 5, 10, 20),
    alpha=1e-4,
    feature_k=1,
    feature_s=1,
):
    """Fit the continuous BLI readout by sample-count-normalized ridge."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    design, targets, metadata = build_integral_design(
        states_scaled,
        transformer,
        dt=dt,
        horizons=horizons,
        feature_k=feature_k,
        feature_s=feature_s,
    )
    count = design.shape[1]
    gram = design @ design.T / count
    cross = targets @ design.T / count
    regularized = gram + float(alpha) * np.eye(design.shape[0])
    W_out = np.linalg.solve(regularized, cross.T).T
    if not np.all(np.isfinite(W_out)):
        raise FloatingPointError("Integral ridge produced non-finite weights.")
    metadata = dict(metadata)
    metadata.update(design_condition=float(np.linalg.cond(gram)), num_constraints=int(count))
    return W_out, metadata


def whittaker_initialize(observations, penalty=100.0):
    """Initialize latent states by penalized second differences."""
    observations = np.asarray(observations, dtype=float)
    if (
        observations.ndim != 2
        or len(observations) < 3
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("observations must be a finite [n, d] array with n >= 3.")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive.")
    count = len(observations)
    second_difference = sparse.diags(
        [np.ones(count - 2), -2.0 * np.ones(count - 2), np.ones(count - 2)],
        offsets=[0, 1, 2],
        shape=(count - 2, count),
        format="csc",
    )
    system = sparse.eye(count, format="csc") + float(penalty) * (
        second_difference.T @ second_difference
    )
    latent = np.column_stack(
        [spsolve(system, observations[:, channel]) for channel in range(observations.shape[1])]
    )
    if not np.all(np.isfinite(latent)):
        raise FloatingPointError("Whittaker initialization produced non-finite states.")
    return latent


def _latent_jacobian_sparsity(count, dim, starts, horizons, feature_k=1, feature_s=1):
    observation_rows = count * dim
    smooth_rows = max(count - 2, 0) * dim
    dynamics_rows = len(starts) * dim
    total_rows = observation_rows + smooth_rows + dynamics_rows + dim
    pattern = sparse.lil_matrix((total_rows, count * dim), dtype=int)
    row = 0
    for index in range(count):
        for channel in range(dim):
            pattern[row, index * dim + channel] = 1
            row += 1
    for index in range(count - 2):
        for channel in range(dim):
            for offset in range(3):
                pattern[row, (index + offset) * dim + channel] = 1
            row += 1
    for start, horizon in zip(starts, horizons):
        stop = int(start + horizon)
        dependency_start = int(start) - (int(feature_k) - 1) * int(feature_s)
        for _ in range(dim):
            for index in range(dependency_start, stop + 1):
                for channel in range(dim):
                    pattern[row, index * dim + channel] = 1
            row += 1
    for channel in range(dim):
        pattern[row, (count - 1) * dim + channel] = 1
        row += 1
    return pattern.tocsr()


def optimize_latent_states(
    observations_scaled,
    initial_scaled,
    W_out,
    transformer,
    config,
    noise_scaled,
):
    """Update a complete training latent trajectory with fixed BLI weights."""
    observations_scaled = np.asarray(observations_scaled, dtype=float)
    initial_scaled = np.asarray(initial_scaled, dtype=float)
    noise_scaled = np.asarray(noise_scaled, dtype=float)
    if (
        observations_scaled.ndim != 2
        or observations_scaled.shape != initial_scaled.shape
        or observations_scaled.shape[1] < 1
    ):
        raise ValueError("BLI latent observations and initialization must share shape [n, d].")
    dim = observations_scaled.shape[1]
    if noise_scaled.shape != (dim,) or np.any(noise_scaled <= 0):
        raise ValueError("noise_scaled must contain one positive value per state.")
    _, _, metadata = build_integral_design(
        initial_scaled,
        transformer,
        config.dt,
        config.horizons,
        feature_k=config.feature_k,
        feature_s=config.feature_s,
    )
    starts = metadata["starts"]
    horizons = metadata["horizons"]
    count, dim = observations_scaled.shape
    jacobian_pattern = _latent_jacobian_sparsity(
        count,
        dim,
        starts,
        horizons,
        feature_k=config.feature_k,
        feature_s=config.feature_s,
    )
    noise_row = noise_scaled.reshape(1, dim)

    def residual(flat):
        states = np.asarray(flat, dtype=float).reshape(count, dim)
        design, targets, current_metadata = build_integral_design(
            states,
            transformer,
            config.dt,
            config.horizons,
            feature_k=config.feature_k,
            feature_s=config.feature_s,
        )
        observation_residual = (states - observations_scaled) / noise_row
        smooth_residual = (
            np.sqrt(config.latent_smoothness_weight)
            * np.diff(states, n=2, axis=0)
            / noise_row
        )
        predicted_deltas = (W_out @ design).T
        target_deltas = targets.T
        horizon_scale = np.sqrt(current_metadata["horizons"]).reshape(-1, 1)
        dynamics_residual = (
            np.sqrt(config.latent_dynamics_weight)
            * (target_deltas - predicted_deltas)
            / (horizon_scale * noise_row)
        )
        endpoint_residual = (
            np.sqrt(config.endpoint_weight)
            * (states[-1] - observations_scaled[-1])
            / noise_scaled
        )
        values = np.concatenate(
            [
                observation_residual.ravel(),
                smooth_residual.ravel(),
                dynamics_residual.ravel(),
                endpoint_residual,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("BLI latent objective became non-finite.")
        return values

    margin = config.latent_bound_margin_std * noise_scaled
    lower = np.min(observations_scaled, axis=0) - margin
    upper = np.max(observations_scaled, axis=0) + margin
    result = least_squares(
        residual,
        np.clip(initial_scaled, lower, upper).ravel(),
        bounds=(np.tile(lower, count), np.tile(upper, count)),
        loss="soft_l1",
        f_scale=1.0,
        jac_sparsity=jacobian_pattern,
        tr_solver="lsmr",
        max_nfev=int(config.latent_max_nfev),
        xtol=config.objective_tolerance,
        ftol=config.objective_tolerance,
        gtol=config.objective_tolerance,
    )
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError("BLI latent solve produced non-finite states.")
    return result.x.reshape(count, dim), result


def _fit_bli_fixed_alpha(observations, config, alpha, method=None):
    # print(f"[BLI ENTER] alpha={alpha}, method={method}, n_obs={len(observations)}", flush=True)
    observation_noise_std = estimate_observation_noise_std(observations)
    latent_original = whittaker_initialize(observations, penalty=config.whittaker_penalty)
    # KEY FIX 1: fit the robust scaler on the raw observations, not on the
    # smoothed latent states. Deployment (MPC / ESO) always transforms
    # observed states, so the training and deployment paths must share the
    # same coordinate system.
    scaler = RobustStateScaler().fit(observations)
    # print(
    #     f"[SCALER-obs]    center_={scaler.center_}  scale_={scaler.scale_}",
    #     flush=True,
    # )
    # latent_scaler_diag = RobustStateScaler().fit(latent_original)
    # print(
    #     f"[SCALER-latent] center_={latent_scaler_diag.center_}  scale_={latent_scaler_diag.scale_}",
    #     flush=True,
    # )
    observations_scaled = scaler.transform(observations)
    latent_scaled = scaler.transform(latent_original)
    noise_scaled = observation_noise_std / scaler.scale_
    objective_history = []
    if method is None:
        method = "bli_delay" if int(config.feature_k) > 1 else "bli"

    for _ in range(int(config.alternating_iterations)):
        embedded_latent, _ = build_causal_embedding(
            latent_scaled, k=config.feature_k, s=config.feature_s
        )
        embedded_observed, _ = build_causal_embedding(
            observations_scaled, k=config.feature_k, s=config.feature_s
        )
        # KEY FIX 2: fit the feature guard envelope on the union of latent
        # and observed embeddings. This keeps deployment-time observations
        # inside the training envelope, so the tanh guard never compresses
        # the closed-loop rate field.
        transformer = QuadraticFeatureTransformer(
            extrapolation_margin=config.extrapolation_margin
        ).fit(np.vstack([embedded_latent, embedded_observed]))
        W_out, integral_metadata = fit_integral_readout(
            latent_scaled,
            transformer,
            dt=config.dt,
            horizons=config.horizons,
            alpha=alpha,
            feature_k=config.feature_k,
            feature_s=config.feature_s,
        )
        candidate, optimization = optimize_latent_states(
            observations_scaled,
            latent_scaled,
            W_out,
            transformer,
            config,
            noise_scaled,
        )
        objective = float(2.0 * optimization.cost)
        objective_history.append(objective)
        latent_scaled = candidate
        if (
            len(objective_history) > 1
            and abs(objective_history[-2] - objective_history[-1])
            <= config.objective_tolerance * max(abs(objective_history[-2]), 1.0)
        ):
            break

    embedded_latent, _ = build_causal_embedding(
        latent_scaled, k=config.feature_k, s=config.feature_s
    )
    embedded_observed, _ = build_causal_embedding(
        observations_scaled, k=config.feature_k, s=config.feature_s
    )
    transformer = QuadraticFeatureTransformer(
        extrapolation_margin=config.extrapolation_margin
    ).fit(np.vstack([embedded_latent, embedded_observed]))
    W_out, integral_metadata = fit_integral_readout(
        latent_scaled,
        transformer,
        dt=config.dt,
        horizons=config.horizons,
        alpha=alpha,
        feature_k=config.feature_k,
        feature_s=config.feature_s,
    )
    return {
        "method": str(method),
        "scaler": scaler,
        "feature_transformer": transformer,
        "W_out": W_out,
        "latent_states": scaler.inverse_transform(latent_scaled),
        "latent_states_scaled": latent_scaled,
        "X_input": scaler.inverse_transform(latent_scaled),
        "objective_history": np.asarray(objective_history),
        "integral_metadata": integral_metadata,
        "selected_alpha": float(alpha),
        "num_features": int(transformer.num_features_),
        "feature_signature": transformer.signature(),
        "feature_k": int(config.feature_k),
        "feature_s": int(config.feature_s),
        "k": int(config.feature_k),
        "s": int(config.feature_s),
        "identification_dt": float(config.dt),
        "observation_noise_std": observation_noise_std,
        "observation_noise_source": "estimated_from_observations",
        "observation_noise_estimator": "second_difference_mad",
        "config": config,
    }


def euler_step_scaled(W_out, transformer, state_scaled, dt, additive_scaled=None):
    """Advance the continuous BLI readout by one forward-Euler step."""
    W_out = np.asarray(W_out, dtype=float)
    state_scaled = np.asarray(state_scaled, dtype=float)
    dim = getattr(transformer, "state_dim_", None)
    feature_count = getattr(transformer, "num_features_", None)
    if dim is None or state_scaled.shape != (dim,) or W_out.shape != (dim, feature_count):
        raise ValueError("BLI Euler dimensions must match the fitted transformer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if additive_scaled is None:
        additive_scaled = np.zeros(dim)
    additive_scaled = np.asarray(additive_scaled, dtype=float)
    if additive_scaled.shape != (dim,) or not np.all(np.isfinite(additive_scaled)):
        raise ValueError("additive_scaled must contain one finite value per state.")

    rate = W_out @ transformer.transform_one(state_scaled) + additive_scaled
    result = state_scaled + float(dt) * rate
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("BLI Euler propagation produced a non-finite state.")
    return result


def rollout_embedded_scaled(
    W_out,
    transformer,
    initial_scaled_history,
    steps,
    dt,
    feature_k=1,
    feature_s=1,
    max_abs_state=1e3,
    additive_scaled=None,
):
    """Roll out a continuous BLI readout from a causal state history."""
    W_out = np.asarray(W_out, dtype=float)
    history_array = np.asarray(initial_scaled_history, dtype=float)
    if history_array.ndim == 1:
        history_array = history_array.reshape(1, -1)
    if (
        history_array.ndim != 2
        or history_array.shape[1] < 1
        or len(history_array) == 0
        or not np.all(np.isfinite(history_array))
    ):
        raise ValueError("BLI initial history must be a non-empty finite [n, d] array.")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    if not isinstance(feature_k, (int, np.integer)) or feature_k <= 0:
        raise ValueError("feature_k must be a positive integer.")
    if not isinstance(feature_s, (int, np.integer)) or feature_s <= 0:
        raise ValueError("feature_s must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    required = (int(feature_k) - 1) * int(feature_s) + 1
    if len(history_array) < required:
        raise ValueError("BLI initial history is shorter than its embedding requirement.")
    physical_dim = int(history_array.shape[1])
    feature_dim = getattr(transformer, "state_dim_", None)
    feature_count = getattr(transformer, "num_features_", None)
    if (
        feature_dim != int(feature_k) * physical_dim
        or W_out.shape != (physical_dim, feature_count)
    ):
        raise ValueError("BLI embedded rollout dimensions do not match the fitted model.")
    if additive_scaled is None:
        additive_scaled = np.zeros(physical_dim, dtype=float)
    additive_scaled = np.asarray(additive_scaled, dtype=float)
    if additive_scaled.shape != (physical_dim,) or not np.all(np.isfinite(additive_scaled)):
        raise ValueError("additive_scaled must contain one finite value per state.")

    history = [row.copy() for row in history_array]
    states = np.empty((int(steps) + 1, physical_dim), dtype=float)
    states[0] = history[-1]
    for index in range(int(steps)):
        current = history[-1]
        if np.max(np.abs(current)) > max_abs_state:
            states[index + 1 :] = np.nan
            break
        embedded = np.concatenate(
            [history[-1 - lag * int(feature_s)] for lag in range(int(feature_k))]
        )
        rate = W_out @ transformer.transform_one(embedded) + additive_scaled
        candidate = current + float(dt) * rate
        if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > max_abs_state:
            states[index + 1 :] = np.nan
            break
        states[index + 1] = candidate
        history.append(candidate)
    return states


def rollout_scaled(
    W_out,
    transformer,
    initial_scaled,
    steps,
    dt,
    max_abs_state=1e3,
    additive_scaled=None,
):
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    states = [np.asarray(initial_scaled, dtype=float).copy()]
    for _ in range(int(steps)):
        next_state = euler_step_scaled(
            W_out,
            transformer,
            states[-1],
            dt,
            additive_scaled=additive_scaled,
        )
        if np.max(np.abs(next_state)) > max_abs_state:
            return np.full((steps + 1, states[0].size), np.nan)
        states.append(next_state)
    return np.asarray(states)


def _build_bli_temporal_folds(observation_count, config):
    fractions = np.asarray(config.validation_fit_fractions, dtype=float)
    fit_counts = np.asarray(
        [int(round(float(fraction) * int(observation_count))) for fraction in fractions],
        dtype=int,
    )
    validation_ends = np.asarray([*fit_counts[1:], int(observation_count)], dtype=int)
    delay = (int(config.feature_k) - 1) * int(config.feature_s)
    fold_anchor_indices = []
    for fit_count, validation_end in zip(fit_counts, validation_ends):
        if int(fit_count) <= delay + max(config.horizons):
            raise ValueError("A BLI validation prefix is too short for its horizons.")
        last_anchor = int(validation_end) - int(config.validation_horizon) - 1
        if last_anchor < int(fit_count):
            raise ValueError("A BLI validation block is too short for its rollout horizon.")
        fold_anchor_indices.append(
            np.unique(
                np.linspace(
                    int(fit_count),
                    last_anchor,
                    int(config.validation_anchors_per_fold),
                    dtype=int,
                )
            )
        )
    return fit_counts, validation_ends, tuple(fold_anchor_indices)


def select_bli_alpha_from_fold_scores(alpha_grid, fold_scores):
    """Select the alpha with the minimum mean temporal-validation score."""
    alphas = np.asarray(alpha_grid, dtype=float)
    scores = np.asarray(fold_scores, dtype=float)
    if (
        alphas.ndim != 1
        or len(alphas) == 0
        or not np.all(np.isfinite(alphas))
        or np.any(alphas <= 0.0)
    ):
        raise ValueError("alpha_grid must contain positive finite values.")
    if (
        scores.ndim != 2
        or scores.shape[0] != len(alphas)
        or scores.shape[1] == 0
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError('fold_scores must contain one finite row per alpha and at least one fold.')

    aggregate = np.mean(scores, axis=1)
    best_index = int(np.argmin(aggregate))
    return {
        "selected_alpha": float(alphas[best_index]),
        "best_alpha": float(alphas[best_index]),
        "selected_index": best_index,
        "best_index": best_index,
        "aggregate_scores": aggregate,
        "selection_rule": "minimum_mean_validation_error",
    }


def select_bli_alpha(observations, config):
    """Select ridge strength on shared multi-anchor temporal folds."""
    observations = np.asarray(observations, dtype=float)
    alphas = np.asarray(config.alpha_grid, dtype=float)
    fit_counts, validation_ends, fold_anchor_indices = _build_bli_temporal_folds(
        len(observations), config
    )
    scores = np.zeros((len(alphas), len(fit_counts)), dtype=float)
    fold_failure_rates = np.zeros_like(scores)
    fold_finite_means = np.full_like(scores, np.inf)
    delay = (int(config.feature_k) - 1) * int(config.feature_s)
    for fold_index, fit_count in enumerate(fit_counts):
        fit_observations = observations[:fit_count]
        latent = whittaker_initialize(fit_observations, penalty=config.whittaker_penalty)
        # KEY FIX 3: same coordinate-system alignment as _fit_bli_fixed_alpha.
        # Fit the scaler on observations and the guard envelope on the union
        # of latent and observed embeddings, so that the validation rollout
        # (which starts from observed history) is scored under the same
        # conditions as the deployed controller.
        scaler = RobustStateScaler().fit(fit_observations)
        latent_scaled = scaler.transform(latent)
        observations_scaled = scaler.transform(fit_observations)
        embedded_latent, _ = build_causal_embedding(
            latent_scaled, k=config.feature_k, s=config.feature_s
        )
        embedded_observed, _ = build_causal_embedding(
            observations_scaled, k=config.feature_k, s=config.feature_s
        )
        transformer = QuadraticFeatureTransformer(
            extrapolation_margin=config.extrapolation_margin
        ).fit(np.vstack([embedded_latent, embedded_observed]))
        for alpha_index, alpha in enumerate(alphas):
            W_out, _ = fit_integral_readout(
                latent_scaled,
                transformer,
                dt=config.dt,
                horizons=config.horizons,
                alpha=float(alpha),
                feature_k=config.feature_k,
                feature_s=config.feature_s,
            )
            rollout_scores = []
            for anchor in fold_anchor_indices[fold_index]:
                history_start = int(anchor) - delay
                initial_scaled = scaler.transform(observations[history_start:int(anchor) + 1])
                prediction_scaled = rollout_embedded_scaled(
                    W_out,
                    transformer,
                    initial_scaled,
                    steps=config.validation_horizon,
                    dt=config.dt,
                    feature_k=config.feature_k,
                    feature_s=config.feature_s,
                )[1:]
                if not np.all(np.isfinite(prediction_scaled)):
                    rollout_scores.append(np.inf)
                    continue
                prediction = scaler.inverse_transform(prediction_scaled)
                truth = observations[
                    int(anchor) + 1 : int(anchor) + config.validation_horizon + 1
                ]
                rollout_scores.append(float(np.sqrt(np.mean((prediction - truth) ** 2))))
            rollout_scores = np.asarray(rollout_scores, dtype=float)
            finite = rollout_scores[np.isfinite(rollout_scores)]
            failure_rate = float(1.0 - len(finite) / len(rollout_scores))
            finite_mean = float(np.mean(finite)) if len(finite) else np.inf
            fold_failure_rates[alpha_index, fold_index] = failure_rate
            fold_finite_means[alpha_index, fold_index] = finite_mean
            scores[alpha_index, fold_index] = (
                finite_mean + config.divergence_penalty * failure_rate
                if np.isfinite(finite_mean)
                else config.divergence_penalty
            )
    selection = select_bli_alpha_from_fold_scores(alphas, scores)
    return {
        **selection,
        "fold_scores": scores,
        "fold_failure_rates": fold_failure_rates,
        "fold_finite_means": fold_finite_means,
        "fold_fit_counts": fit_counts,
        "fold_validation_ends": validation_ends,
        "fold_anchor_indices": fold_anchor_indices,
    }


def train_bli_from_observations(t_observed, X_observed, config=None, method='bli'):
    """Train BLI strictly from the declared observation prefix."""
    if config is None:
        config = BLIConfig()
    config.validate()
    t_observed = np.asarray(t_observed, dtype=float)
    X_observed = np.asarray(X_observed, dtype=float)
    required = int(config.training_samples) + 1
    if t_observed.ndim != 1 or X_observed.ndim != 2 or X_observed.shape[1] < 1:
        raise ValueError("BLI observations must have time [n] and state [n, d].")
    if len(t_observed) != len(X_observed) or len(t_observed) < required:
        raise ValueError("BLI observations are shorter than the declared training window.")
    t_used = t_observed[:required].copy()
    X_used = X_observed[:required].copy()
    if not np.all(np.isfinite(t_used)) or not np.all(np.isfinite(X_used)):
        raise ValueError("BLI training observations must be finite.")
    if not np.allclose(np.diff(t_used), config.dt, rtol=0.0, atol=1e-12):
        raise ValueError("BLI training observations must use config.dt.")
    validation = select_bli_alpha(X_used[: config.training_samples], config)
    alpha = validation["selected_alpha"]
    model = _fit_bli_fixed_alpha(X_used, config, alpha, method=method)
    model.update(
        t_observed=t_used,
        X_observed=X_used,
        train_len=config.training_samples,
        handoff_time=float(t_used[-1]),
        handoff_observation=X_used[-1].copy(),
        validation=validation,
        best_alpha=validation["best_alpha"],
        alpha_selection_rule=validation["selection_rule"],
    )
    return model
