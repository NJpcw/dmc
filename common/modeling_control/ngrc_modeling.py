from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from . import latent_integral_modeling as _BLI


DEFAULT_COMPARISON_METHODS = ("raw", "raw_delay", "bli", "bli_delay")

# 1. Configuration and example system


@dataclass
class ExperimentConfig:
    x0: np.ndarray = field(default_factory=lambda: np.array([7.5, 12.5]))
    t_end: float = 100.0
    dt_original: float = 0.001
    stride: int = 10
    noise_std: float = 0.05
    random_seed: int = 1

    training_duration: float = 3.0
    evaluation_duration: float = 5.0

    k: int = 2
    s: int = 1
    poly_order: int = 2
    feature_clip: float = 50.0
    max_abs_state: float = 1e3

    sg_window: int = 40
    sg_degree: int = 2
    multi_step_q: int = 5

    bli_horizons: tuple = (1, 2, 5, 10, 20)
    bli_alpha_grid: tuple = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    bli_alternating_iterations: int = 3
    bli_whittaker_penalty: float = 100.0
    bli_latent_smoothness_weight: float = 0.2
    bli_latent_dynamics_weight: float = 1.0
    bli_endpoint_weight: float = 4.0
    bli_latent_bound_margin_std: float = 6.0
    bli_latent_max_nfev: int = 12
    bli_objective_tolerance: float = 1e-4
    bli_extrapolation_margin: float = 2.0
    validation_horizon: int = 20
    validation_anchors_per_fold: int = 12
    fit_fractions: tuple = (0.40, 0.60, 0.80)
    divergence_penalty: float = 1e3
    alpha_grid: np.ndarray = field(default_factory=lambda: np.logspace(-4, 2, 80))

    external_horizon: int = 20
    external_anchors: int = 60

    @property
    def identification_dt(self):
        return self.dt_original * self.stride

    @property
    def training_samples(self):
        return int(round(self.training_duration / self.identification_dt))

    def validate(self):
        if self.t_end <= 0 or self.dt_original <= 0:
            raise ValueError("t_end and dt_original must be positive.")
        if not isinstance(self.stride, (int, np.integer)) or self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if self.training_duration <= 0 or self.evaluation_duration <= 0:
            raise ValueError("Training and evaluation durations must be positive.")
        required_end = (
            self.training_duration
            + self.evaluation_duration
            + self.external_horizon * self.identification_dt
        )
        if self.t_end <= required_end:
            raise ValueError(
                "t_end must extend beyond the evaluation window by one rollout horizon."
            )
        if (
            not isinstance(self.k, (int, np.integer))
            or self.k <= 0
            or not isinstance(self.s, (int, np.integer))
            or self.s <= 0
        ):
            raise ValueError("k and s must be positive integers.")
        if self.poly_order not in (1, 2, 3):
            raise ValueError("poly_order must be 1, 2, or 3.")
        if self.sg_window <= self.sg_degree or self.sg_degree < 0:
            raise ValueError("SG window must be larger than its non-negative degree.")
        if self.multi_step_q <= 0:
            raise ValueError("multi_step_q must be positive.")
        if self.validation_horizon <= 0 or self.external_horizon <= 0:
            raise ValueError("Rollout horizons must be positive.")
        if self.validation_anchors_per_fold <= 0 or self.external_anchors <= 0:
            raise ValueError("Anchor counts must be positive.")
        alpha_grid = np.asarray(self.alpha_grid, dtype=float)
        if alpha_grid.ndim != 1 or len(alpha_grid) == 0:
            raise ValueError("alpha_grid must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(alpha_grid)) or np.any(alpha_grid <= 0):
            raise ValueError("Every alpha must be finite and positive.")
        _make_bli_config(self).validate()


def _make_bli_config(config, feature_k=1, feature_s=1):
    return _BLI.BLIConfig(
        training_samples=config.training_samples,
        dt=config.identification_dt,
        horizons=tuple(config.bli_horizons),
        alpha_grid=tuple(config.bli_alpha_grid),
        alternating_iterations=config.bli_alternating_iterations,
        whittaker_penalty=config.bli_whittaker_penalty,
        latent_smoothness_weight=config.bli_latent_smoothness_weight,
        latent_dynamics_weight=config.bli_latent_dynamics_weight,
        endpoint_weight=config.bli_endpoint_weight,
        latent_bound_margin_std=config.bli_latent_bound_margin_std,
        latent_max_nfev=config.bli_latent_max_nfev,
        objective_tolerance=config.bli_objective_tolerance,
        extrapolation_margin=config.bli_extrapolation_margin,
        validation_fit_fractions=tuple(config.fit_fractions),
        validation_horizon=config.validation_horizon,
        validation_anchors_per_fold=config.validation_anchors_per_fold,
        divergence_penalty=config.divergence_penalty,
        feature_k=int(feature_k),
        feature_s=int(feature_s),
    )


def grassland_abnormal_system(state, t):
    x, y = state
    r = 0.3
    K = 50.0
    alpha = 0.03
    beta = 0.8
    d = 0.3

    dx = r * x * (1.0 - x / K) - alpha * x * y
    dy = beta * alpha * x * y - d * y
    return [dx, dy]


def euler_step_grassland(state, t, dt):
    if dt <= 0:
        raise ValueError("dt must be positive.")
    state = np.asarray(state, dtype=float)
    rate = np.asarray(grassland_abnormal_system(state, t), dtype=float)
    return np.maximum(state + dt * rate, 0.0)


def generate_data(
    x0=np.array([7.5, 12.5]),
    t_end=100.0,
    dt=0.001,
    noise_std=0.05,
    random_seed=1,
    stride=1,
):
    if t_end <= 0 or dt <= 0:
        raise ValueError("t_end and dt must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative.")
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("stride must be a positive integer.")

    t = np.arange(0.0, t_end, dt)
    X_clean = np.empty((len(t), 2), dtype=float)
    X_clean[0] = np.asarray(x0, dtype=float)
    for index in range(len(t) - 1):
        X_clean[index + 1] = euler_step_grassland(X_clean[index], t[index], dt)
    sample_indices = np.arange(0, len(t), stride, dtype=np.int64)
    t = t[sample_indices]
    X_clean = X_clean[sample_indices]
    rng = np.random.RandomState(random_seed)
    X_noisy = X_clean + noise_std * rng.randn(*X_clean.shape)
    return t, X_clean, X_noisy


# 2. Scaling and causal SG filtering


class StandardScalerNP:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("Scaler input must be a non-empty 2-D array.")
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, Xs):
        return np.asarray(Xs, dtype=float) * self.std_ + self.mean_


def _causal_savgol_weights(length, degree):
    tau = np.arange(-length + 1, 1, dtype=float)
    degree_eff = min(degree, length - 1)
    V = np.vander(tau, N=degree_eff + 1, increasing=True)
    return np.linalg.pinv(V)[0]


def causal_savgol_filter(X, window=9, degree=2):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a non-negative integer.")
    if degree >= window:
        raise ValueError("degree must be smaller than window.")

    T = len(X)
    X_hat = np.zeros_like(X)
    weight_cache = {
        length: _causal_savgol_weights(length, degree)
        for length in range(1, window + 1)
    }
    for i in range(T):
        length = min(window, i + 1)
        X_hat[i] = weight_cache[length] @ X[i - length + 1 : i + 1]
    return X_hat


def make_feature_from_olin(o_lin, poly_order=2, feature_clip=50.0):
    o_lin = np.asarray(o_lin, dtype=float)
    o_lin = np.clip(o_lin, -feature_clip, feature_clip)
    values = [1.0]
    values.extend(o_lin)

    if poly_order >= 2:
        for a in range(len(o_lin)):
            for b in range(a, len(o_lin)):
                values.append(o_lin[a] * o_lin[b])

    if poly_order >= 3:
        for a in range(len(o_lin)):
            for b in range(a, len(o_lin)):
                for c in range(b, len(o_lin)):
                    values.append(o_lin[a] * o_lin[b] * o_lin[c])

    return np.asarray(values, dtype=float)


def build_ngrc_features(Xs, k=2, s=1, poly_order=2, feature_clip=50.0):
    Xs = np.asarray(Xs, dtype=float)
    if Xs.ndim != 2:
        raise ValueError("Xs must be a 2-D array.")
    if k <= 0 or s <= 0:
        raise ValueError("k and s must be positive.")

    start_index = (k - 1) * s
    if len(Xs) <= start_index:
        raise ValueError("Data are too short for the selected k and s.")

    valid_indices = np.arange(start_index, len(Xs))
    features = []
    for i in valid_indices:
        o_lin = np.concatenate([Xs[i - delay * s] for delay in range(k)])
        features.append(make_feature_from_olin(o_lin, poly_order, feature_clip))
    return np.asarray(features).T, valid_indices


def build_single_feature(Xs_history, k=2, s=1, poly_order=2, feature_clip=50.0):
    Xs_history = np.asarray(Xs_history, dtype=float)
    min_history = (k - 1) * s + 1
    if Xs_history.ndim != 2 or len(Xs_history) < min_history:
        raise ValueError("Not enough 2-D history for NGRC feature construction.")

    current = len(Xs_history) - 1
    o_lin = np.concatenate([Xs_history[current - delay * s] for delay in range(k)])
    return make_feature_from_olin(
        o_lin,
        poly_order=poly_order,
        feature_clip=feature_clip,
    ).reshape(-1, 1)


def make_training_matrices(
    Xs_train,
    k=2,
    s=1,
    poly_order=2,
    q=1,
    average_increment=True,
    feature_clip=50.0,
):
    if q <= 0:
        raise ValueError("q must be positive.")
    O_total, valid_indices = build_ngrc_features(
        Xs_train,
        k=k,
        s=s,
        poly_order=poly_order,
        feature_clip=feature_clip,
    )
    mask = valid_indices + q < len(Xs_train)
    valid_indices = valid_indices[mask]
    O_total = O_total[:, mask]
    if O_total.shape[1] == 0:
        raise ValueError("No valid training pairs; increase data length or reduce q.")

    if average_increment:
        Yd = (Xs_train[valid_indices + q] - Xs_train[valid_indices]) / q
    else:
        Yd = Xs_train[valid_indices + q] - Xs_train[valid_indices]
    return O_total, Yd


def train_ngrc_ridge_normalized(O_total, Yd, alpha=1e-3):
    O = np.asarray(O_total, dtype=float)
    Yd = np.asarray(Yd, dtype=float)
    if O.ndim != 2 or Yd.ndim != 2 or O.shape[1] != Yd.shape[0]:
        raise ValueError("O_total and Yd have incompatible shapes.")
    if O.shape[1] == 0:
        raise ValueError("At least one training sample is required.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")

    n_samples = O.shape[1]
    A = (O @ O.T) / n_samples + alpha * np.eye(O.shape[0])
    B = (O @ Yd) / n_samples
    try:
        return np.linalg.solve(A, B).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0].T


def build_raw_embedding(states_scaled, k=1, s=1):
    """Build causal Raw-NGRC state taps ordered from current to oldest."""
    states_scaled = np.asarray(states_scaled, dtype=float)
    if (
        states_scaled.ndim != 2
        or states_scaled.shape[1] < 1
        or len(states_scaled) == 0
        or not np.all(np.isfinite(states_scaled))
    ):
        raise ValueError("Raw embedding states must be a non-empty finite [n, d] array.")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("Raw embedding k must be a positive integer.")
    if not isinstance(s, (int, np.integer)) or s <= 0:
        raise ValueError("Raw embedding s must be a positive integer.")

    start_index = (int(k) - 1) * int(s)
    if len(states_scaled) <= start_index:
        raise ValueError("Raw observations are too short for the requested embedding.")
    valid_indices = np.arange(start_index, len(states_scaled), dtype=int)
    embedded = np.asarray(
        [
            np.concatenate([states_scaled[index - delay * int(s)] for delay in range(int(k))])
            for index in valid_indices
        ],
        dtype=float,
    )
    return embedded, valid_indices


def fit_raw_ngrc(
    observations,
    alpha=1e-3,
    identification_dt=0.01,
    extrapolation_margin=2.0,
    k=1,
    s=1,
    method="raw",
):
    """Fit one-step Raw-NGRC with a guarded quadratic causal embedding."""
    observations = np.asarray(observations, dtype=float)
    if (
        observations.ndim != 2
        or observations.shape[1] < 1
        or len(observations) < 2
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("Raw observations must be a finite [n, d] array with n >= 2.")
    if not np.isfinite(identification_dt) or identification_dt <= 0:
        raise ValueError("identification_dt must be finite and positive.")
    scaler = _BLI.RobustStateScaler().fit(observations)
    scaled = scaler.transform(observations)
    embedded, valid_indices = build_raw_embedding(scaled, k=k, s=s)
    pair_mask = valid_indices + 1 < len(scaled)
    if not np.any(pair_mask):
        raise ValueError("Raw observations contain no target after the embedding history.")
    transformer = _BLI.QuadraticFeatureTransformer(
        extrapolation_margin=extrapolation_margin
    ).fit(embedded)
    design = transformer.transform(embedded[pair_mask]).T
    target_indices = valid_indices[pair_mask]
    targets = scaled[target_indices + 1] - scaled[target_indices]
    W_out = train_ngrc_ridge_normalized(design, targets, alpha=alpha)
    gram = design @ design.T / design.shape[1]
    return {
        "method": str(method),
        "X_input": observations.copy(),
        "scaler": scaler,
        "feature_transformer": transformer,
        "feature_signature": transformer.signature(),
        "W_out": W_out,
        "selected_alpha": float(alpha),
        "weight_norm": float(np.linalg.norm(W_out)),
        "feature_condition": float(np.linalg.cond(gram)),
        "num_features": int(transformer.num_features_),
        "fit_sample_count": int(len(observations)),
        "k": int(k),
        "s": int(s),
        "poly_order": 2,
        "identification_dt": float(identification_dt),
    }


def ngrc_autonomous_predict_scaled(
    Xs_initial_history,
    W_out,
    pred_steps,
    k=2,
    s=1,
    poly_order=2,
    feature_clip=50.0,
    max_abs_state=1e3,
):
    Xs_initial_history = np.asarray(Xs_initial_history, dtype=float)
    if pred_steps < 0:
        raise ValueError("pred_steps cannot be negative.")

    total_len = len(Xs_initial_history) + pred_steps
    Xs_pred = np.zeros((total_len, Xs_initial_history.shape[1]))
    Xs_pred[: len(Xs_initial_history)] = Xs_initial_history

    for pos in range(len(Xs_initial_history) - 1, total_len - 1):
        if not np.all(np.isfinite(Xs_pred[pos])):
            Xs_pred[pos + 1 :] = np.nan
            break
        if np.max(np.abs(Xs_pred[pos])) > max_abs_state:
            Xs_pred[pos + 1 :] = np.nan
            break

        feature = build_single_feature(
            Xs_pred[: pos + 1],
            k=k,
            s=s,
            poly_order=poly_order,
            feature_clip=feature_clip,
        )
        delta = (W_out @ feature).ravel()
        if not np.all(np.isfinite(delta)):
            Xs_pred[pos + 1 :] = np.nan
            break
        Xs_pred[pos + 1] = Xs_pred[pos] + delta

    return Xs_pred


def raw_shared_autonomous_predict_scaled(
    initial_scaled_history,
    W_out,
    feature_transformer,
    pred_steps,
    k=1,
    s=1,
    max_abs_state=1e3,
):
    """Roll out a guarded Raw-NGRC discrete increment map causally."""
    initial_scaled_history = np.asarray(initial_scaled_history, dtype=float)
    W_out = np.asarray(W_out, dtype=float)
    if initial_scaled_history.ndim == 1:
        initial_scaled_history = initial_scaled_history.reshape(1, -1)
    if (
        initial_scaled_history.ndim != 2
        or initial_scaled_history.shape[1] < 1
        or not np.all(np.isfinite(initial_scaled_history))
    ):
        raise ValueError("Raw initial history must be a non-empty finite [n, d] array.")
    if not isinstance(pred_steps, (int, np.integer)) or pred_steps < 0:
        raise ValueError("pred_steps must be a nonnegative integer.")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("Raw rollout k must be a positive integer.")
    if not isinstance(s, (int, np.integer)) or s <= 0:
        raise ValueError("Raw rollout s must be a positive integer.")
    required = (int(k) - 1) * int(s) + 1
    if len(initial_scaled_history) < required:
        raise ValueError("Raw initial history is shorter than its embedding requirement.")

    history = [row.copy() for row in initial_scaled_history]
    states = np.empty((int(pred_steps) + 1, initial_scaled_history.shape[1]), dtype=float)
    states[0] = history[-1]
    for index in range(int(pred_steps)):
        current = history[-1]
        if not np.all(np.isfinite(current)) or np.max(np.abs(current)) > max_abs_state:
            states[index + 1 :] = np.nan
            break
        embedded = np.concatenate([history[-1 - delay * int(s)] for delay in range(int(k))])
        feature = feature_transformer.transform_one(embedded)
        increment = W_out @ feature
        candidate = current + increment
        if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate)) > max_abs_state:
            states[index + 1 :] = np.nan
            break
        states[index + 1] = candidate
        history.append(candidate)
    return states


def predict_one_step_original(model, history_original):
    history_original = np.asarray(history_original, dtype=float)
    scaler = model["scaler"]
    scaled_history = scaler.transform(history_original)
    if model.get("method") in ("bli", "bli_delay"):
        feature_k = int(model.get("k", 1))
        feature_s = int(model.get("s", 1))
        required = (feature_k - 1) * feature_s + 1
        if len(scaled_history) < required:
            raise ValueError("BLI history is shorter than its embedding requirement.")
        next_scaled = _BLI.rollout_embedded_scaled(
            model["W_out"],
            model["feature_transformer"],
            scaled_history[-required:],
            steps=1,
            dt=model["identification_dt"],
            feature_k=feature_k,
            feature_s=feature_s,
        )[1]
        return scaler.inverse_transform(next_scaled.reshape(1, -1))[0]
    if model.get("method") in ("raw", "raw_delay") and "feature_transformer" in model:
        embedded, _ = build_raw_embedding(scaled_history, k=int(model['k']), s=int(model['s']))
        feature = model["feature_transformer"].transform_one(embedded[-1])
    else:
        feature = build_single_feature(
            scaled_history,
            k=model["k"],
            s=model["s"],
            poly_order=model["poly_order"],
            feature_clip=model["feature_clip"],
        )
    increment = (np.asarray(model["W_out"], dtype=float) @ feature).ravel()
    next_scaled = scaled_history[-1] + increment
    return scaler.inverse_transform(next_scaled.reshape(1, -1))[0]


# 4. Statistics and rolling-origin validation


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.size == 0 or y_true.shape != y_pred.shape:
        return np.inf
    if not np.all(np.isfinite(y_pred)):
        return np.inf
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def summarize_rmse_values(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    total = len(values)
    stat = {
        "num_total": total,
        "num_finite": len(finite),
        "failure_rate": 1.0 - len(finite) / total if total else np.nan,
    }
    if len(finite):
        stat.update(
            finite_mean=float(np.mean(finite)),
            finite_std=float(np.std(finite)),
            finite_max=float(np.max(finite)),
            finite_min=float(np.min(finite)),
        )
    else:
        stat.update(finite_mean=np.inf, finite_std=np.nan, finite_max=np.inf, finite_min=np.inf)
    return stat


@dataclass(frozen=True)
class TemporalFold:
    fit_end: int
    val_start: int
    val_end: int

    def last_anchor(self, horizon):
        return self.val_end - horizon - 1


def build_temporal_folds(n_samples, horizon, fit_fractions=(0.5, 0.65, 0.8)):
    if n_samples <= 0 or horizon <= 0:
        raise ValueError("n_samples and horizon must be positive.")
    fractions = np.asarray(fit_fractions, dtype=float)
    if fractions.ndim != 1 or len(fractions) == 0:
        raise ValueError("fit_fractions must be a non-empty sequence.")
    if np.any(fractions <= 0) or np.any(fractions >= 1):
        raise ValueError("Every fit fraction must lie strictly between zero and one.")
    if np.any(np.diff(fractions) <= 0):
        raise ValueError("fit_fractions must be strictly increasing.")

    fit_ends = [int(round(fraction * n_samples)) for fraction in fractions]
    val_ends = fit_ends[1:] + [n_samples]
    folds = []
    for fit_end, val_end in zip(fit_ends, val_ends):
        fold = TemporalFold(fit_end=fit_end, val_start=fit_end, val_end=val_end)
        if fold.last_anchor(horizon) < fold.val_start:
            raise ValueError("A validation block is too short for the rollout horizon.")
        folds.append(fold)
    return folds


@dataclass(frozen=True)
class AlphaSelection:
    selected_alpha: float
    best_alpha: float
    selected_index: int
    best_index: int
    aggregate_scores: np.ndarray
    fold_scores: np.ndarray


def select_minimum_mean_alpha(alpha_grid, fold_scores):
    alphas = np.asarray(alpha_grid, dtype=float)
    scores = np.asarray(fold_scores, dtype=float)
    if (
        alphas.ndim != 1
        or len(alphas) == 0
        or not np.all(np.isfinite(alphas))
        or np.any(alphas <= 0.0)
    ):
        raise ValueError("alpha_grid must contain positive finite values.")
    if scores.ndim != 2 or scores.shape[0] != len(alphas):
        raise ValueError("fold_scores must have shape [num_alphas, num_folds].")
    if scores.shape[1] == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("fold_scores must contain finite scores for every fold.")

    aggregate = np.mean(scores, axis=1)
    selected_index = int(np.argmin(aggregate))

    return AlphaSelection(
        selected_alpha=float(alphas[selected_index]),
        best_alpha=float(alphas[selected_index]),
        selected_index=selected_index,
        best_index=selected_index,
        aggregate_scores=aggregate,
        fold_scores=scores,
    )


def _anchors_for_fold(fold, horizon, num_anchors, min_anchor):
    first = max(fold.val_start, min_anchor)
    last = fold.last_anchor(horizon)
    if last < first:
        raise ValueError("No valid rollout anchor exists in a validation fold.")
    return np.unique(np.linspace(first, last, num_anchors, dtype=int))


def select_alpha_by_rolling_validation(X_train, q, config):
    X_train = np.asarray(X_train, dtype=float)
    folds = build_temporal_folds(
        n_samples=len(X_train),
        horizon=config.validation_horizon,
        fit_fractions=config.fit_fractions,
    )
    alphas = np.sort(np.asarray(config.alpha_grid, dtype=float))
    fold_scores = np.zeros((len(alphas), len(folds)))
    fold_failure_rates = np.zeros_like(fold_scores)
    fold_finite_means = np.full_like(fold_scores, np.inf)

    min_history = (config.k - 1) * config.s + 1

    for fold_index, fold in enumerate(folds):
        scaler = StandardScalerNP().fit(X_train[: fold.fit_end])
        Xs_until_validation_end = scaler.transform(X_train[: fold.val_end])
        Xs_fit = Xs_until_validation_end[: fold.fit_end]
        O_fit, Y_fit = make_training_matrices(
            Xs_fit,
            k=config.k,
            s=config.s,
            poly_order=config.poly_order,
            q=q,
            average_increment=True,
            feature_clip=config.feature_clip,
        )
        anchors = _anchors_for_fold(
            fold,
            horizon=config.validation_horizon,
            num_anchors=config.validation_anchors_per_fold,
            min_anchor=min_history - 1,
        )

        for alpha_index, alpha in enumerate(alphas):
            W = train_ngrc_ridge_normalized(O_fit, Y_fit, alpha=alpha)
            rmse_values = []
            for anchor in anchors:
                prediction = ngrc_autonomous_predict_scaled(
                    Xs_initial_history=Xs_until_validation_end[: anchor + 1],
                    W_out=W,
                    pred_steps=config.validation_horizon,
                    k=config.k,
                    s=config.s,
                    poly_order=config.poly_order,
                    feature_clip=config.feature_clip,
                    max_abs_state=config.max_abs_state,
                )
                predicted_window = prediction[
                    anchor + 1 : anchor + config.validation_horizon + 1
                ]
                true_window = Xs_until_validation_end[
                    anchor + 1 : anchor + config.validation_horizon + 1
                ]
                rmse_values.append(safe_rmse(true_window, predicted_window))

            stat = summarize_rmse_values(rmse_values)
            fold_failure_rates[alpha_index, fold_index] = stat["failure_rate"]
            fold_finite_means[alpha_index, fold_index] = stat["finite_mean"]
            if np.isfinite(stat["finite_mean"]):
                fold_scores[alpha_index, fold_index] = (
                    stat["finite_mean"]
                    + config.divergence_penalty * stat["failure_rate"]
                )
            else:
                fold_scores[alpha_index, fold_index] = config.divergence_penalty

    selection = select_minimum_mean_alpha(alphas, fold_scores)
    return {
        "selection": selection,
        "alpha_grid": alphas,
        "folds": folds,
        "fold_scores": fold_scores,
        "fold_failure_rates": fold_failure_rates,
        "fold_finite_means": fold_finite_means,
    }


def select_raw_alpha_by_rolling_validation(X_train, config, k=1, s=1):
    """Select Raw ridge strength for a guarded causal state embedding."""
    X_train = np.asarray(X_train, dtype=float)
    folds = build_temporal_folds(
        n_samples=len(X_train),
        horizon=config.validation_horizon,
        fit_fractions=config.fit_fractions,
    )
    alphas = np.sort(np.asarray(config.alpha_grid, dtype=float))
    fold_scores = np.zeros((len(alphas), len(folds)))
    fold_failure_rates = np.zeros_like(fold_scores)
    fold_finite_means = np.full_like(fold_scores, np.inf)
    for fold_index, fold in enumerate(folds):
        fit_original = X_train[: fold.fit_end]
        scaler = _BLI.RobustStateScaler().fit(fit_original)
        scaled = scaler.transform(X_train[: fold.val_end])
        fit_scaled = scaled[: fold.fit_end]
        fit_embedded, fit_indices = build_raw_embedding(fit_scaled, k=k, s=s)
        fit_pair_mask = fit_indices + 1 < len(fit_scaled)
        if not np.any(fit_pair_mask):
            raise ValueError("A Raw validation prefix has no embedded target pair.")
        transformer = _BLI.QuadraticFeatureTransformer(
            extrapolation_margin=config.bli_extrapolation_margin
        ).fit(fit_embedded)
        design = transformer.transform(fit_embedded[fit_pair_mask]).T
        target_indices = fit_indices[fit_pair_mask]
        targets = fit_scaled[target_indices + 1] - fit_scaled[target_indices]
        anchors = _anchors_for_fold(
            fold,
            horizon=config.validation_horizon,
            num_anchors=config.validation_anchors_per_fold,
            min_anchor=(int(k) - 1) * int(s),
        )
        for alpha_index, alpha in enumerate(alphas):
            weights = train_ngrc_ridge_normalized(design, targets, alpha=alpha)
            values = []
            for anchor in anchors:
                prediction = raw_shared_autonomous_predict_scaled(
                    scaled[anchor - (int(k) - 1) * int(s) : anchor + 1],
                    weights,
                    transformer,
                    pred_steps=config.validation_horizon,
                    k=k,
                    s=s,
                    max_abs_state=config.max_abs_state,
                )[1:]
                prediction_original = scaler.inverse_transform(prediction)
                truth_original = X_train[
                    anchor + 1 : anchor + config.validation_horizon + 1
                ]
                values.append(safe_rmse(truth_original, prediction_original))
            stat = summarize_rmse_values(values)
            fold_failure_rates[alpha_index, fold_index] = stat["failure_rate"]
            fold_finite_means[alpha_index, fold_index] = stat["finite_mean"]
            fold_scores[alpha_index, fold_index] = (
                stat["finite_mean"] + config.divergence_penalty * stat["failure_rate"]
                if np.isfinite(stat["finite_mean"])
                else config.divergence_penalty
            )
    selection = select_minimum_mean_alpha(alphas, fold_scores)
    return {
        "selection": selection,
        "alpha_grid": alphas,
        "folds": folds,
        "fold_scores": fold_scores,
        "fold_failure_rates": fold_failure_rates,
        "fold_finite_means": fold_finite_means,
    }


# 5. Method training and fixed-window external evaluation


def _method_input_and_q(method, X_noisy, X_sg, config):
    if method == "raw":
        return X_noisy, 1
    if method == "sg":
        return X_sg, 1
    if method == "sg_ms":
        return X_sg, config.multi_step_q
    raise ValueError("method must be 'raw', 'sg', or 'sg_ms'.")


def train_bli_method(t_observed, X_observed, config, method='bli', feature_k=1, feature_s=1):
    """Fit BLI on the declared prefix, then causally filter later observations."""
    t_observed = np.asarray(t_observed, dtype=float)
    X_observed = np.asarray(X_observed, dtype=float)
    bli_config = _make_bli_config(config, feature_k=feature_k, feature_s=feature_s)
    model = _BLI.train_bli_from_observations(
        t_observed,
        X_observed,
        config=bli_config,
        method=method,
    )

    validation_raw = model["validation"]
    alphas = np.asarray(bli_config.alpha_grid, dtype=float)
    selected_index = int(np.flatnonzero(alphas == model["selected_alpha"])[0])
    best_index = int(np.flatnonzero(alphas == model["best_alpha"])[0])
    fold_scores = np.asarray(validation_raw["fold_scores"], dtype=float)
    aggregate_scores = np.asarray(validation_raw['aggregate_scores'], dtype=float)
    selection = AlphaSelection(
        selected_alpha=float(model["selected_alpha"]),
        best_alpha=float(model["best_alpha"]),
        selected_index=selected_index,
        best_index=best_index,
        aggregate_scores=aggregate_scores,
        fold_scores=fold_scores,
    )
    model["validation"] = {
        "selection": selection,
        "alpha_grid": alphas,
        "fold_scores": fold_scores,
        "selection_rule": str(validation_raw["selection_rule"]),
        "raw": validation_raw,
    }

    required = config.training_samples + 1
    training_states = np.asarray(model["latent_states"], dtype=float)
    if len(X_observed) > required:
        model_input = np.vstack([training_states, X_observed[required:]])
    else:
        model_input = training_states.copy()
    model.update(
        X_input=model_input,
        X_observed=X_observed.copy(),
        q=tuple(sorted(bli_config.horizons)),
        train_len=config.training_samples,
        weight_norm=float(np.linalg.norm(model["W_out"])),
        feature_condition=float(model["integral_metadata"]["design_condition"]),
        k=int(feature_k),
        s=int(feature_s),
        poly_order=2,
        feature_clip=config.feature_clip,
        identification_dt=config.identification_dt,
        config=config,
        bli_config=bli_config,
        fit_sample_count=int(len(model["X_observed"])),
    )
    return model


def _raw_method_embedding(method, config):
    if method == "raw":
        return 1, 1
    if method == "raw_delay":
        if int(config.k) < 2:
            raise ValueError("raw_delay requires at least two taps.")
        return int(config.k), int(config.s)
    raise ValueError("Raw method must be 'raw' or 'raw_delay'.")


def _bli_method_embedding(method, config):
    if method == "bli":
        return 1, 1
    if method == "bli_delay":
        if int(config.k) < 2:
            raise ValueError("bli_delay requires at least two taps.")
        return int(config.k), int(config.s)
    raise ValueError("BLI method must be 'bli' or 'bli_delay'.")


def train_one_method(method, X_noisy, X_sg, config):
    if method not in ("raw", "raw_delay", "bli", "bli_delay", "sg", "sg_ms"):
        raise ValueError(f"Unsupported modeling method: {method}.")
    if method in ("bli", "bli_delay"):
        feature_k, feature_s = _bli_method_embedding(method, config)
        t_observed = np.arange(len(X_noisy), dtype=float) * config.identification_dt
        return train_bli_method(
            t_observed,
            X_noisy,
            config,
            method=method,
            feature_k=feature_k,
            feature_s=feature_s,
        )
    train_len = config.training_samples
    if train_len >= len(X_noisy):
        raise ValueError("Training duration leaves no samples for evaluation.")

    if method in ("raw", "raw_delay"):
        raw_k, raw_s = _raw_method_embedding(method, config)
        X_train = X_noisy[:train_len]
        X_fit = X_noisy[: train_len + 1]
        validation = select_raw_alpha_by_rolling_validation(X_train, config, k=raw_k, s=raw_s)
        selection = validation["selection"]
        result = fit_raw_ngrc(
            X_fit,
            alpha=selection.selected_alpha,
            identification_dt=config.identification_dt,
            extrapolation_margin=config.bli_extrapolation_margin,
            k=raw_k,
            s=raw_s,
            method=method,
        )
        result.update(
            X_input=X_noisy,
            q=1,
            train_len=train_len,
            best_alpha=selection.best_alpha,
            validation=validation,
            feature_clip=config.feature_clip,
            config=config,
        )
        return result

    X_input, q = _method_input_and_q(method, X_noisy, X_sg, config)
    X_train = X_input[:train_len]
    validation = select_alpha_by_rolling_validation(X_train, q=q, config=config)
    selection = validation["selection"]

    scaler = StandardScalerNP().fit(X_train)
    Xs_train = scaler.transform(X_train)
    O_total, Yd = make_training_matrices(
        Xs_train,
        k=config.k,
        s=config.s,
        poly_order=config.poly_order,
        q=q,
        average_increment=True,
        feature_clip=config.feature_clip,
    )
    W_out = train_ngrc_ridge_normalized(O_total, Yd, alpha=selection.selected_alpha)
    gram = O_total @ O_total.T / O_total.shape[1]

    return {
        "method": method,
        "X_input": X_input,
        "q": q,
        "train_len": train_len,
        "scaler": scaler,
        "W_out": W_out,
        "selected_alpha": selection.selected_alpha,
        "best_alpha": selection.best_alpha,
        "validation": validation,
        "weight_norm": float(np.linalg.norm(W_out)),
        "feature_condition": float(np.linalg.cond(gram)),
        "num_features": int(O_total.shape[0]),
        "k": config.k,
        "s": config.s,
        "poly_order": config.poly_order,
        "feature_clip": config.feature_clip,
    }


def train_method_from_observations(t_observed, X_observed, config, method="bli"):
    config.validate()
    if method not in ("raw", "raw_delay", "bli", "bli_delay", "sg", "sg_ms"):
        raise ValueError(f"Unsupported modeling method: {method}.")
    t_observed = np.asarray(t_observed, dtype=float)
    X_observed = np.asarray(X_observed, dtype=float)
    if t_observed.ndim != 1 or X_observed.ndim != 2:
        raise ValueError("Observed times and states must be one- and two-dimensional.")
    if len(t_observed) != len(X_observed):
        raise ValueError("Observed times and states must have equal length.")
    required = config.training_samples + 1
    if len(t_observed) < required:
        raise ValueError("Post-alarm observations must include training and handoff samples.")
    if not np.all(np.isfinite(t_observed[:required])) or not np.all(
        np.isfinite(X_observed[:required])
    ):
        raise ValueError("Post-alarm observations must be finite.")
    if not np.allclose(
        np.diff(t_observed[:required]),
        config.identification_dt,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Post-alarm observations must use identification_dt.")

    t_used = t_observed[:required].copy()
    X_used = X_observed[:required].copy()
    if method in ("bli", "bli_delay"):
        feature_k, feature_s = _bli_method_embedding(method, config)
        return train_bli_method(
            t_used,
            X_used,
            config,
            method=method,
            feature_k=feature_k,
            feature_s=feature_s,
        )
    if method in ("raw", "raw_delay"):
        raw_k, raw_s = _raw_method_embedding(method, config)
        X_train = X_used[: config.training_samples]
        validation = select_raw_alpha_by_rolling_validation(X_train, config, k=raw_k, s=raw_s)
        selection = validation["selection"]
        result = fit_raw_ngrc(
            X_used,
            alpha=selection.selected_alpha,
            identification_dt=config.identification_dt,
            extrapolation_margin=config.bli_extrapolation_margin,
            k=raw_k,
            s=raw_s,
            method=method,
        )
        result.update(
            X_observed=X_used,
            X_input=X_used,
            t_observed=t_used,
            q=1,
            train_len=config.training_samples,
            best_alpha=selection.best_alpha,
            validation=validation,
            feature_clip=config.feature_clip,
            handoff_time=float(t_used[-1]),
            handoff_observation=X_used[-1].copy(),
            config=config,
        )
        return result
    X_sg = causal_savgol_filter(X_used, window=config.sg_window, degree=config.sg_degree)
    X_input, q = _method_input_and_q(method, X_used, X_sg, config)
    X_train = X_input[: config.training_samples]
    validation = select_alpha_by_rolling_validation(X_train, q=q, config=config)
    selection = validation["selection"]

    scaler = StandardScalerNP().fit(X_train)
    Xs_train = scaler.transform(X_train)
    O_total, Yd = make_training_matrices(
        Xs_train,
        k=config.k,
        s=config.s,
        poly_order=config.poly_order,
        q=q,
        average_increment=True,
        feature_clip=config.feature_clip,
    )
    W_out = train_ngrc_ridge_normalized(O_total, Yd, alpha=selection.selected_alpha)
    gram = O_total @ O_total.T / O_total.shape[1]
    return {
        "method": method,
        "X_observed": X_used,
        "X_input": X_input,
        "t_observed": t_used,
        "q": q,
        "train_len": config.training_samples,
        "scaler": scaler,
        "W_out": W_out,
        "selected_alpha": selection.selected_alpha,
        "best_alpha": selection.best_alpha,
        "validation": validation,
        "weight_norm": float(np.linalg.norm(W_out)),
        "feature_condition": float(np.linalg.cond(gram)),
        "num_features": int(O_total.shape[0]),
        "k": config.k,
        "s": config.s,
        "poly_order": config.poly_order,
        "feature_clip": config.feature_clip,
        "identification_dt": config.identification_dt,
        "handoff_time": float(t_used[-1]),
        "handoff_observation": X_used[-1].copy(),
        "config": config,
    }


def rollout_from_anchor(result, X_clean, anchor_index, horizon, max_abs_state=1e3):
    if anchor_index + horizon >= len(X_clean):
        raise ValueError("The requested rollout exceeds the available data.")

    scaler = result["scaler"]
    if result.get("method") in ("bli", "bli_delay"):
        feature_k = int(result.get("k", 1))
        feature_s = int(result.get("s", 1))
        history_start = anchor_index - (feature_k - 1) * feature_s
        if history_start < 0:
            raise ValueError("The BLI rollout anchor has insufficient embedding history.")
        initial_scaled = scaler.transform(result['X_input'][history_start:anchor_index + 1])
        prediction_scaled = _BLI.rollout_embedded_scaled(
            result["W_out"],
            result["feature_transformer"],
            initial_scaled,
            steps=horizon,
            dt=result["identification_dt"],
            feature_k=feature_k,
            feature_s=feature_s,
            max_abs_state=max_abs_state,
        )
        predicted_window = scaler.inverse_transform(prediction_scaled)
        true_window = X_clean[anchor_index : anchor_index + horizon + 1]
        rmse = safe_rmse(true_window[1:], predicted_window[1:])
        return true_window, predicted_window, rmse
    if result.get("method") in ("raw", "raw_delay") and "feature_transformer" in result:
        raw_k = int(result["k"])
        raw_s = int(result["s"])
        history_start = anchor_index - (raw_k - 1) * raw_s
        if history_start < 0:
            raise ValueError("The Raw rollout anchor has insufficient embedding history.")
        initial_scaled = scaler.transform(result['X_input'][history_start:anchor_index + 1])
        prediction_scaled = raw_shared_autonomous_predict_scaled(
            initial_scaled,
            result["W_out"],
            result["feature_transformer"],
            pred_steps=horizon,
            k=raw_k,
            s=raw_s,
            max_abs_state=max_abs_state,
        )
        predicted_window = scaler.inverse_transform(prediction_scaled)
        true_window = X_clean[anchor_index : anchor_index + horizon + 1]
        rmse = safe_rmse(true_window[1:], predicted_window[1:])
        return true_window, predicted_window, rmse
    Xs_history = scaler.transform(result["X_input"][: anchor_index + 1])
    Xs_prediction = ngrc_autonomous_predict_scaled(
        Xs_initial_history=Xs_history,
        W_out=result["W_out"],
        pred_steps=horizon,
        k=result["k"],
        s=result["s"],
        poly_order=result["poly_order"],
        feature_clip=result["feature_clip"],
        max_abs_state=max_abs_state,
    )
    prediction = scaler.inverse_transform(Xs_prediction)
    predicted_window = prediction[anchor_index : anchor_index + horizon + 1]
    true_window = X_clean[anchor_index : anchor_index + horizon + 1]
    rmse = safe_rmse(true_window[1:], predicted_window[1:])
    return true_window, predicted_window, rmse


def get_fixed_window_anchors(t, start_time, end_time, horizon, num_anchors):
    first = int(np.searchsorted(t, start_time, side="left"))
    last = min(len(t) - horizon - 1, int(np.searchsorted(t, end_time, side='right')) - 1)
    if last < first:
        raise ValueError("The external evaluation window contains no valid anchor.")
    return np.unique(np.linspace(first, last, num_anchors, dtype=int))


def evaluate_external_window(results, config, methods):
    evaluation_start = float(results.get('evaluation_start_time', config.training_duration))
    anchors = get_fixed_window_anchors(
        t=results["t"],
        start_time=evaluation_start,
        end_time=evaluation_start + config.evaluation_duration,
        horizon=config.external_horizon,
        num_anchors=config.external_anchors,
    )

    for method in methods:
        values = []
        for anchor in anchors:
            _, _, rmse = rollout_from_anchor(
                results[method],
                results["X_clean"],
                anchor,
                config.external_horizon,
                max_abs_state=config.max_abs_state,
            )
            values.append(rmse)
        results[method]["external_rmse_values"] = np.asarray(values)
        results[method]["external_stat"] = summarize_rmse_values(values)

    results["external_anchors"] = anchors
    return results


def run_experiment(config=None, methods=DEFAULT_COMPARISON_METHODS, make_plots=True, verbose=True):
    if config is None:
        config = ExperimentConfig()
    config.validate()

    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one modeling method is required.")

    t, X_clean, X_noisy = generate_data(
        x0=config.x0,
        t_end=config.t_end,
        dt=config.dt_original,
        noise_std=config.noise_std,
        random_seed=config.random_seed,
        stride=config.stride,
    )
    t_id = t
    X_clean_id = X_clean
    X_noisy_id = X_noisy
    X_sg_id = causal_savgol_filter(X_noisy_id, window=config.sg_window, degree=config.sg_degree)

    results = {
        "config": config,
        "t": t_id,
        "X_clean": X_clean_id,
        "X_noisy": X_noisy_id,
        "X_sg": X_sg_id,
        "methods": methods,
    }

    for method in methods:
        results[method] = train_one_method(
            method=method,
            X_noisy=X_noisy_id,
            X_sg=X_sg_id,
            config=config,
        )

    evaluate_external_window(results, config, methods)

    if verbose:
        print_experiment_summary(results)
    if make_plots:
        plot_alpha_validation(results)
        plot_rollout_examples(results)
        plot_external_rmse(results)
    return results


# 6. Reporting and plots


def print_experiment_summary(results):
    config = results["config"]
    methods = results["methods"]
    anchors = results["external_anchors"]
    t = results["t"]

    print("========== NGRC modeling comparison ==========")
    print(f"Identification sampling interval: {config.identification_dt:.4f}")
    print(f"Training duration: {config.training_duration:.2f} s")
    print(f"Training samples: {config.training_samples}")
    print(f'External evaluation window: {t[anchors[0]]:.4f} to {t[anchors[-1]]:.4f}')
    print("Normalized ridge alpha (sample-count invariant form):")
    for method in methods:
        result = results[method]
        selection_text = f"selected empirical minimum = {result['selected_alpha']:.8g}"
        print(
            f"{method:8s} | {selection_text} | "
            f"||W||_F = {result['weight_norm']:.6f} | "
            f"features = {result['num_features']} | "
            f"cond = {result['feature_condition']:.6g}"
        )

    print("------------------------------------------------")
    print(f"External rollout horizon N = {config.external_horizon}")
    for method in methods:
        result = results[method]
        stat = result["external_stat"]
        selected_index = result["validation"]["selection"].selected_index
        fold_scores = result["validation"]["fold_scores"][selected_index]
        print(
            f"{method:8s} | alpha = {result['selected_alpha']:<11.6g} | "
            f"fold scores = {np.array2string(fold_scores, precision=4)} | "
            f"mean RMSE = {stat['finite_mean']:.6f} | "
            f"std = {stat['finite_std']:.6f} | "
            f"max = {stat['finite_max']:.6f} | "
            f"failure = {100.0 * stat['failure_rate']:.2f}%"
        )
    print("================================================")


def plot_alpha_validation(results, output_path=None, show=True):
    methods = results["methods"]
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, 3.2 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        result = results[method]
        validation = result["validation"]
        selection = validation["selection"]
        alphas = validation["alpha_grid"]
        scores = selection.aggregate_scores.copy()
        scores[~np.isfinite(scores)] = np.nan

        ax = axes[row, 0]
        ax.semilogx(alphas, scores, linewidth=2, label="Mean rolling score")
        ax.axvline(
            selection.selected_alpha,
            linestyle=":",
            linewidth=1.5,
            label="Selected empirical minimum",
        )
        ax.set_title(f"{method.upper()} robust alpha validation")
        ax.set_xlabel("Normalized ridge alpha")
        ax.set_ylabel("Penalized rolling score")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_rollout_examples(results, num_anchors=6, output_path=None, show=True):
    config = results["config"]
    methods = results["methods"]
    clean_states = np.asarray(results["X_clean"], dtype=float)
    if clean_states.ndim != 2 or clean_states.shape[1] < 1:
        raise ValueError("X_clean must contain at least one state channel.")
    state_dim = clean_states.shape[1]
    default_state_names = ("x", "y") if state_dim == 2 else tuple(
        f"x{index + 1}" for index in range(state_dim)
    )
    state_names = tuple(results.get("state_names", default_state_names))
    if len(state_names) != state_dim:
        raise ValueError("state_names must contain one label per state channel.")
    all_anchors = results["external_anchors"]
    positions = np.unique(np.linspace(0, len(all_anchors) - 1, num_anchors, dtype=int))
    anchors = all_anchors[positions]
    method_labels = {
        "raw": "Current-state Raw-NGRC (k=1)",
        "raw_delay": "Delay-embedded Raw-NGRC (k=2, s=1)",
        "sg_ms": "SG-MS-NGRC",
        "sg": "Fixed SG-NGRC",
        "bli": "BLI-NGRC (k=1)",
        "bli_delay": "Delay-embedded BLI-NGRC (k=2, s=1)",
    }
    method_colors = {
        "raw": "#4C78A8",
        "raw_delay": "#54A24B",
        "sg_ms": "#E07A5F",
        "sg": "#59A14F",
        "bli": "#F28E2B",
        "bli_delay": "#E15759",
    }

    fig, axes = plt.subplots(
        len(methods),
        state_dim,
        figsize=(5.5 * state_dim, 3.2 * len(methods)),
        squeeze=False,
        sharex=False,
    )
    for row, method in enumerate(methods):
        prediction_color = method_colors.get(method, "#4C78A8")
        for anchor_id, anchor in enumerate(anchors):
            true_window, predicted_window, _ = rollout_from_anchor(
                results[method],
                results["X_clean"],
                anchor,
                config.external_horizon,
                max_abs_state=config.max_abs_state,
            )
            t_window = results["t"][
                anchor : anchor + config.external_horizon + 1
            ]
            for dim in range(state_dim):
                ax = axes[row, dim]
                ax.plot(
                    t_window,
                    true_window[:, dim],
                    color="#222222",
                    linewidth=1.8,
                    label="Clean trajectory" if anchor_id == 0 else None,
                )
                ax.plot(
                    t_window,
                    predicted_window[:, dim],
                    "--",
                    color=prediction_color,
                    linewidth=1.6,
                    label="Autonomous prediction" if anchor_id == 0 else None,
                )
                ax.set_title(
                    f"{method_labels.get(method, method.upper())} — "
                    f"{state_names[dim]} ({config.external_horizon}-step)"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(state_names[dim])
                ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if anchor_id == 0:
                    ax.legend(frameon=False, loc="best")
    fig.suptitle("Rolling-origin prediction after NGRC modeling", y=1.01)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_external_rmse(results, output_path=None, show=True):
    methods = results["methods"]
    anchors = results["external_anchors"]
    t = results["t"]
    config = results["config"]

    fig = plt.figure(figsize=(10, 4.8))
    for method in methods:
        values = results[method]["external_rmse_values"].copy()
        values[~np.isfinite(values)] = np.nan
        plt.plot(t[anchors], values, marker='o', linewidth=1.8, label=method.upper())
    plt.xlabel("Rollout starting time")
    plt.ylabel(f"{config.external_horizon}-step rollout RMSE")
    plt.title(
        "Fixed-window external NGRC rollout error "
        f"[{t[anchors[0]]:.2f}, {t[anchors[-1]]:.2f}]"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    run_experiment(config=ExperimentConfig(), make_plots=True, verbose=True)
