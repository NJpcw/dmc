"""Training, checkpoint selection, detection, and result generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .residual_glr_detection import (
    GLRCalibration,
    GLRDetectionResult,
    MGCRLSTM,
    Normalization,
    PostAlarmSegment,
    build_post_alarm_segment,
    calibrate_glr,
    detect_switch,
    fit_normalization,
    generate_normal_trajectories,
    make_windows,
    resolve_state_noise,
    simulate_sampled_trajectory,
    simulate_trajectory,
)


CHECKPOINT_VERSION = 3
PREDICTOR_FIELDS = (
    "state_dim",
    "noise_std",
    "input_window",
    "conv_channels",
    "conv_kernel_size",
    "hidden_size",
    "lstm_layers",
    "dropout",
    "sample_dt",
    "simulation_integrator",
)


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/predator_prey/detection")
    checkpoint_epoch: int | None = None
    seed: int = 2026
    t_end: float = 200.0
    switch_time: float = 100.0
    dt: float = 0.001
    stride: int = 50
    state_dim: int = 2
    noise_std: float | np.ndarray = 0.05
    initial_x1: float = 12.19
    initial_x2: float = 24.4
    training_trajectories: int = 12
    validation_trajectories: int = 4
    trajectory_duration: float = 80.0
    input_window: int = 12
    glr_window: int = 20
    glr_min_segment: int = 3
    glr_threshold_quantile: float = 0.97
    covariance_regularization: float = 1e-6
    conv_channels: int = 16
    conv_kernel_size: int = 3
    hidden_size: int = 32
    lstm_layers: int = 2
    dropout: float = 0.0
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"

    @property
    def observation_dt(self) -> float:
        return self.dt * self.stride

    def validate(self) -> None:
        if self.t_end <= 0 or self.dt <= 0 or self.trajectory_duration <= 0:
            raise ValueError("time settings must be positive")
        if not isinstance(self.stride, (int, np.integer)) or self.stride < 1:
            raise ValueError("stride must be a positive integer")
        if not 0.0 < self.switch_time < self.t_end:
            raise ValueError("switch_time must lie strictly inside (0, t_end)")
        if not isinstance(self.state_dim, (int, np.integer)) or self.state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        noise_std = resolve_state_noise(self.noise_std, self.state_dim)
        if np.any(noise_std <= 0):
            raise ValueError("noise_std must be positive")
        if self.training_trajectories < 1 or self.validation_trajectories < 1:
            raise ValueError("training and validation trajectory counts must be positive")
        if self.input_window < 2:
            raise ValueError("input_window must be at least 2")
        if self.glr_min_segment < 2 or self.glr_window < self.glr_min_segment:
            raise ValueError("GLR window settings are invalid")
        if not 0.0 < self.glr_threshold_quantile <= 1.0:
            raise ValueError("glr_threshold_quantile must lie inside (0, 1]")
        if not np.isfinite(self.covariance_regularization) or self.covariance_regularization <= 0:
            raise ValueError("covariance_regularization must be finite and positive")
        if min(
            self.conv_channels,
            self.conv_kernel_size,
            self.hidden_size,
            self.lstm_layers,
            self.batch_size,
        ) < 1:
            raise ValueError("network dimensions and batch_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie inside [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.checkpoint_epoch is None:
            if self.epochs < 1:
                raise ValueError("epochs must be positive")
        elif self.checkpoint_epoch < 10 or self.checkpoint_epoch % 10:
            raise ValueError("checkpoint_epoch must be None or a positive multiple of 10")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    def checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"epoch_{epoch:04d}.pt"


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float


@dataclass(frozen=True)
class PredictorResult:
    model: MGCRLSTM
    normalization: Normalization
    best_epoch: int
    best_validation_loss: float
    source: str
    validation_trajectories: list[np.ndarray] | None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def predictor_signature(config: ExperimentConfig) -> dict[str, object]:
    return {
        "state_dim": config.state_dim,
        "noise_std": resolve_state_noise(config.noise_std, config.state_dim).tolist(),
        "input_window": config.input_window,
        "conv_channels": config.conv_channels,
        "conv_kernel_size": config.conv_kernel_size,
        "hidden_size": config.hidden_size,
        "lstm_layers": config.lstm_layers,
        "dropout": config.dropout,
        "sample_dt": config.observation_dt,
        "simulation_integrator": "forward_euler",
    }


def build_model(config: ExperimentConfig) -> MGCRLSTM:
    return MGCRLSTM(
        state_dim=config.state_dim,
        conv_channels=config.conv_channels,
        kernel_size=config.conv_kernel_size,
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    )


def combine_windows(
    trajectories: list[np.ndarray],
    input_window: int,
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    windows = [make_windows(item, input_window, normalization) for item in trajectories]
    return (
        np.concatenate([item[0] for item in windows]),
        np.concatenate([item[1] for item in windows]),
    )


def _validate_checkpoint(
    payload: object,
    config: ExperimentConfig,
    expected_epoch: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    required = {
        "checkpoint_version",
        "epoch",
        "model_state_dict",
        "train_loss",
        "validation_loss",
        "normalization",
        "predictor_signature",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
    if int(payload["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if int(payload["epoch"]) != expected_epoch:
        raise ValueError(f"checkpoint contains epoch {payload['epoch']}, expected {expected_epoch}")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("model_state_dict must contain model tensors")

    saved_signature = payload["predictor_signature"]
    requested_signature = predictor_signature(config)
    if not isinstance(saved_signature, dict):
        raise ValueError("predictor_signature must be a dictionary")
    mismatches = [
        name
        for name in PREDICTOR_FIELDS
        if saved_signature.get(name) != requested_signature[name]
    ]
    if mismatches:
        details = ", ".join(
            f"{name}={saved_signature.get(name)!r} (saved) != "
            f"{requested_signature[name]!r} (configured)"
            for name in mismatches
        )
        raise ValueError(f"checkpoint is incompatible: {details}")

    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or not {"mean", "std"} <= normalization.keys():
        raise ValueError("checkpoint normalization must contain mean and std")
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    expected_shape = (config.state_dim,)
    if mean.shape != expected_shape or std.shape != expected_shape or np.any(std <= 0):
        raise ValueError("checkpoint normalization is invalid")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("checkpoint normalization must be finite")
    for name in ("train_loss", "validation_loss"):
        if not np.isfinite(float(payload[name])):
            raise ValueError(f"checkpoint {name} must be finite")
    return dict(payload)


def save_checkpoint_atomic(
    path: Path,
    model: MGCRLSTM,
    normalization: Normalization,
    config: ExperimentConfig,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "normalization": normalization.as_dict(),
        "predictor_signature": predictor_signature(config),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        saved = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_checkpoint(saved, config, epoch)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path,
    config: ExperimentConfig,
    device: torch.device,
    expected_epoch: int,
) -> PredictorResult:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = _validate_checkpoint(
        torch.load(path, map_location=device, weights_only=True),
        config,
        expected_epoch,
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    normalization = Normalization(
        mean=np.asarray(payload["normalization"]["mean"], dtype=np.float64),
        std=np.asarray(payload["normalization"]["std"], dtype=np.float64),
    )
    return PredictorResult(
        model=model,
        normalization=normalization,
        best_epoch=expected_epoch,
        best_validation_loss=float(payload["validation_loss"]),
        source="loaded",
        validation_trajectories=None,
    )


def train_model(
    model: MGCRLSTM,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    normalization: Normalization,
    config: ExperimentConfig,
    device: torch.device,
) -> TrainingResult:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_x = torch.from_numpy(validation_inputs).to(device)
    validation_y = torch.from_numpy(validation_targets).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    rows: list[dict[str, float | int]] = []
    candidates: list[tuple[float, int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_inputs)
            total_samples += len(batch_inputs)

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).cpu())
        train_loss = total_loss / max(total_samples, 1)
        rows.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss})
        if epoch % 10 == 0 or epoch == config.epochs:
            save_checkpoint_atomic(
                config.checkpoint_path(epoch),
                model,
                normalization,
                config,
                epoch,
                train_loss,
                validation_loss,
            )
            candidates.append((validation_loss, epoch))
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:4d}/{config.epochs}: "
                f"train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}"
            )

    best_validation_loss, best_epoch = min(candidates)
    return TrainingResult(pd.DataFrame(rows), best_epoch, best_validation_loss)


def load_or_train_predictor(
    config: ExperimentConfig,
    device: torch.device,
    *,
    save_history: bool = True,
) -> PredictorResult:
    if config.checkpoint_epoch is not None:
        path = config.checkpoint_path(config.checkpoint_epoch)
        print(f"Loading checkpoint: {path}")
        return load_checkpoint(path, config, device, config.checkpoint_epoch)

    print("Generating normal-subsystem training and validation trajectories...")
    training_trajectories = generate_normal_trajectories(
        config.training_trajectories,
        config.trajectory_duration,
        config.dt,
        config.noise_std,
        config.seed + 10,
        config.stride,
    )
    validation_trajectories = generate_normal_trajectories(
        config.validation_trajectories,
        config.trajectory_duration,
        config.dt,
        config.noise_std,
        config.seed + 20_000,
        config.stride,
    )
    normalization = fit_normalization(training_trajectories)
    train_inputs, train_targets = combine_windows(
        training_trajectories, config.input_window, normalization
    )
    validation_inputs, validation_targets = combine_windows(
        validation_trajectories, config.input_window, normalization
    )
    model = build_model(config)
    training = train_model(
        model,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        normalization,
        config,
        device,
    )
    if save_history:
        training.history.to_csv(Path(config.output_dir) / "history.csv", index=False)
    best = load_checkpoint(
        config.checkpoint_path(training.best_epoch), config, device, training.best_epoch
    )
    return PredictorResult(
        model=best.model,
        normalization=best.normalization,
        best_epoch=best.best_epoch,
        best_validation_loss=best.best_validation_loss,
        source="trained",
        validation_trajectories=validation_trajectories,
    )


def predict_trajectory(
    model: MGCRLSTM,
    trajectory: np.ndarray,
    input_window: int,
    normalization: Normalization,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs, _ = make_windows(trajectory, input_window, normalization)
    input_tensor = torch.from_numpy(inputs)
    predicted_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(input_tensor), batch_size):
            batch = input_tensor[start : start + batch_size].to(device)
            predicted_batches.append(model(batch).cpu().numpy())
    physical_predictions = normalization.inverse(np.concatenate(predicted_batches))
    predictions = np.full_like(trajectory, np.nan, dtype=np.float64)
    predictions[input_window:] = physical_predictions
    return predictions, np.asarray(trajectory, dtype=np.float64) - predictions


def calibrate_predictor(
    predictor: PredictorResult,
    config: ExperimentConfig,
    device: torch.device,
) -> GLRCalibration:
    trajectories = predictor.validation_trajectories or generate_normal_trajectories(
        config.validation_trajectories,
        config.trajectory_duration,
        config.dt,
        config.noise_std,
        config.seed + 20_000,
        config.stride,
    )
    residuals = [
        predict_trajectory(
            predictor.model,
            trajectory,
            config.input_window,
            predictor.normalization,
            device,
            config.batch_size,
        )[1]
        for trajectory in trajectories
    ]
    return calibrate_glr(
        residuals,
        config.glr_window,
        config.glr_min_segment,
        config.glr_threshold_quantile,
        config.covariance_regularization,
    )


def save_detection_table(
    path: Path,
    time: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    predictions: np.ndarray,
    innovations: np.ndarray,
    result: GLRDetectionResult,
    calibration: GLRCalibration,
) -> None:
    candidate_time = np.full(len(time), np.nan, dtype=np.float64)
    valid_candidate = (result.candidate_index >= 0) & (result.candidate_index < len(time))
    candidate_time[valid_candidate] = time[result.candidate_index[valid_candidate]]
    pd.DataFrame(
        {
            "time": time,
            "x1_clean": clean[:, 0],
            "x2_clean": clean[:, 1],
            "x1_observed": noisy[:, 0],
            "x2_observed": noisy[:, 1],
            "x1_predicted": predictions[:, 0],
            "x2_predicted": predictions[:, 1],
            "e1": innovations[:, 0],
            "e2": innovations[:, 1],
            "mahalanobis_energy": result.energy,
            "glr_statistic": result.statistic,
            "glr_candidate_index": result.candidate_index,
            "glr_candidate_time": candidate_time,
            "glr_threshold": calibration.threshold,
            "alarm": result.alarm.astype(int),
        }
    ).to_csv(path, index=False)


def plot_experiment(
    path: Path,
    time: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    innovations: np.ndarray,
    result: GLRDetectionResult,
    calibration: GLRCalibration,
    true_switch_time: float,
    alarm_time: float | None,
    estimated_switch_time: float | None,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    axes[0].plot(time, noisy[:, 0], color="tab:green", alpha=0.45, lw=0.8, label=r"Observed $x_1$")
    axes[0].plot(time, noisy[:, 1], color="tab:blue", alpha=0.45, lw=0.8, label=r"Observed $x_2$")
    axes[0].plot(time, clean[:, 0], color="darkgreen", lw=1.5, label=r"Clean $x_1$")
    axes[0].plot(time, clean[:, 1], color="navy", lw=1.5, label=r"Clean $x_2$")
    axes[0].set_ylabel("State")
    axes[0].set_title("MG-CLSTM switch-point detection: grassland-herbivore system")
    axes[1].plot(time, innovations[:, 0], color="tab:green", label=r"$e_1$")
    axes[1].plot(time, innovations[:, 1], color="tab:blue", label=r"$e_2$")
    axes[1].set_ylabel("Signed innovation")
    axes[1].legend()
    axes[2].plot(time, result.energy, color="darkgreen", label=r"$D_t$")
    axes[2].set_ylabel("Mahalanobis energy")
    axes[2].legend()
    axes[3].plot(time, result.statistic, color="purple", label=r"$G_t$")
    axes[3].axhline(
        calibration.threshold,
        color="orange",
        ls="--",
        alpha=0.8,
        label=r"GLR threshold $h$",
    )
    axes[3].set_ylabel("GLR statistic")
    axes[3].set_xlabel("Time")
    axes[3].legend(ncol=2, fontsize=9)
    for axis in axes:
        axis.axvline(true_switch_time, color="black", ls="--", lw=1.4, label="True switch")
        if alarm_time is not None:
            axis.axvline(alarm_time, color="crimson", ls="-.", lw=1.4, label="Alarm")
        if estimated_switch_time is not None:
            axis.axvline(
                estimated_switch_time,
                color="darkorange",
                ls=":",
                lw=1.6,
                label="Estimated switch",
            )
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), ncol=3, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _calibration_summary(calibration: GLRCalibration) -> dict[str, object]:
    return {
        "residual_mean": calibration.residual_mean.tolist(),
        "residual_covariance": calibration.residual_covariance.tolist(),
        "threshold": calibration.threshold,
        "window": calibration.window,
        "min_segment": calibration.min_segment,
        "threshold_quantile": calibration.threshold_quantile,
        "covariance_regularization": calibration.covariance_regularization,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    config.validate()
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    predictor = load_or_train_predictor(config, device)
    print(f"Predictor source: {predictor.source}")
    calibration = calibrate_predictor(predictor, config, device)
    trajectory = simulate_sampled_trajectory(
        initial_state=np.array([config.initial_x1, config.initial_x2]),
        t_end=config.t_end,
        internal_dt=config.dt,
        observation_stride=config.stride,
        noise_std=config.noise_std,
        seed=config.seed + 30_000,
        switch_time=config.switch_time,
    )
    time = trajectory.sample_time
    clean = trajectory.sample_clean
    noisy = trajectory.sample_noisy
    predictions, innovations = predict_trajectory(
        predictor.model,
        noisy,
        config.input_window,
        predictor.normalization,
        device,
        config.batch_size,
    )
    result = detect_switch(innovations, calibration)
    alarm_time = float(time[result.alarm_index]) if result.alarm_index is not None else None
    estimated_switch_time = (
        float(time[result.estimated_switch_index])
        if result.estimated_switch_index is not None
        else None
    )
    alarm_delay = alarm_time - config.switch_time if alarm_time is not None else None
    localization_error = (
        estimated_switch_time - config.switch_time if estimated_switch_time is not None else None
    )
    metrics: dict[str, object] = {
        "seed": config.seed,
        "noise_std": config.noise_std,
        "true_switch_time": config.switch_time,
        "alarm_time": alarm_time,
        "estimated_switch_time": estimated_switch_time,
        "alarm_delay": alarm_delay,
        "localization_error": localization_error,
        "model_source": predictor.source,
        "best_checkpoint_epoch": predictor.best_epoch,
        "best_validation_loss": predictor.best_validation_loss,
        "glr_calibration": _calibration_summary(calibration),
        "normalization": predictor.normalization.as_dict(),
    }
    save_detection_table(
        output_dir / "detection_series.csv",
        time,
        clean,
        noisy,
        predictions,
        innovations,
        result,
        calibration,
    )
    plot_experiment(
        output_dir / "experiment.png",
        time,
        clean,
        noisy,
        innovations,
        result,
        calibration,
        config.switch_time,
        alarm_time,
        estimated_switch_time,
    )

    print("\nExperiment complete")
    print(f"True switch time: {config.switch_time:.3f}")
    if alarm_time is None:
        print("Alarm time: none")
        print("Estimated switch time: none")
    else:
        print(f"Alarm time: {alarm_time:.3f}")
        print(f"Estimated switch time: {estimated_switch_time:.3f}")
        print(f"Alarm delay: {alarm_delay:.3f}")
        print(f"Localization error: {localization_error:.3f}")
    print(
        f"Selected checkpoint: epoch {predictor.best_epoch} "
        f"(validation_loss={predictor.best_validation_loss:.6f})"
    )
    print(f"Results: {output_dir.resolve()}")
    return metrics


def evaluate_detection_trials(
    config, predictor, calibration, device, seeds, initial_state,
    simulate_trajectory, predict, detect,
):
    """Evaluate a fixed detector; only post-switch alarms enter conditional means."""
    false_alarms = 0
    misses = 0
    post_switch_errors: list[float] = []
    post_switch_delays: list[float] = []
    trial_count = len(seeds)

    print(
        "trial_seed  simulation_seed  status             "
        "alarm_time  estimated_switch_time  localization_error  alarm_delay"
    )
    for trial_seed in seeds:
        simulation_seed = config.seed + 30_000 + trial_seed
        trajectory = simulate_trajectory(
            initial_state=initial_state,
            t_end=config.t_end,
            internal_dt=config.dt,
            observation_stride=config.stride,
            noise_std=config.noise_std,
            seed=simulation_seed,
            switch_time=config.switch_time,
        )
        _, innovations = predict(
            predictor.model,
            trajectory.sample_noisy,
            config.input_window,
            predictor.normalization,
            device,
            config.batch_size,
        )
        result = detect(innovations, calibration)
        if result.alarm_index is None:
            status = "miss"
            alarm_time = None
            estimated_time = None
            localization_error = None
            alarm_delay = None
            misses += 1
        else:
            if result.estimated_switch_index is None:
                raise RuntimeError("an alarm has no estimated switch index")
            alarm_time = float(trajectory.sample_time[result.alarm_index])
            estimated_time = float(trajectory.sample_time[result.estimated_switch_index])
            localization_error = abs(estimated_time - config.switch_time)
            alarm_delay = alarm_time - config.switch_time
            if alarm_time < config.switch_time:
                status = "false_alarm"
                false_alarms += 1
            else:
                status = "post_switch_alarm"
                post_switch_errors.append(localization_error)
                post_switch_delays.append(alarm_delay)

        alarm_text = "none" if alarm_time is None else f"{alarm_time:.6f}"
        estimated_text = "none" if estimated_time is None else f"{estimated_time:.6f}"
        error_text = (
            "none" if localization_error is None else f"{localization_error:.6f}"
        )
        delay_text = "none" if alarm_delay is None else f"{alarm_delay:.6f}"
        print(
            f"{trial_seed:10d}  {simulation_seed:15d}  {status:17s}  "
            f"{alarm_text:10s}  {estimated_text:21s}  {error_text}  {delay_text}",
            flush=True,
        )

    mean_error = (
        sum(post_switch_errors) / len(post_switch_errors) if post_switch_errors else None
    )
    mean_delay = (
        sum(post_switch_delays) / len(post_switch_delays) if post_switch_delays else None
    )
    print("\nSummary")
    print(f"trial_count: {trial_count}")
    print(f"training_seed: {config.seed}")
    print(f"checkpoint_epoch: {predictor.best_epoch}")
    print(f"predictor_source: {predictor.source}")
    print(f"validation_loss: {predictor.best_validation_loss:.8g}")
    print(f"glr_threshold_quantile: {config.glr_threshold_quantile}")
    print(f"glr_threshold: {calibration.threshold:.8g}")
    print(f"false_alarm_count: {false_alarms}")
    print(f"false_alarm_rate: {false_alarms / trial_count:.6f}")
    print(f"miss_count: {misses}")
    print(f"miss_rate: {misses / trial_count:.6f}")
    print(
        "mean_alarm_delay_post_switch: "
        + ("undefined" if mean_delay is None else f"{mean_delay:.6f}")
    )
    print(f"post_switch_alarm_count: {len(post_switch_errors)}")
    print(
        "mean_localization_error_post_switch: "
        + ("undefined" if mean_error is None else f"{mean_error:.6f}")
    )
