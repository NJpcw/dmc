"""Three-channel MG-CRLSTM prediction and GLR switch detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.detection import residual_glr_detection as CORE
from common.detection import detector_training as EXPERIMENT

from .dynamics import (
    DEFAULT_INITIAL_STATE,
    DEFAULT_NOISE_STD,
    STATE_DIM,
    SWITCH_TIME,
    generate_normal_trajectories,
    resolve_noise_std,
    simulate_sampled_trajectory,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parent


@dataclass(frozen=True)
class DetectorConfig:
    output_dir: Path = (
        REPOSITORY_ROOT / "outputs/waterwheel/pipeline/detection"
    )
    checkpoint_epoch: int | None = None
    seed: int = 2026
    t_end: float = 110.0
    switch_time: float = SWITCH_TIME
    dt: float = 0.001
    stride: int = 50
    noise_std: np.ndarray = field(default_factory=lambda: DEFAULT_NOISE_STD.copy())
    initial_state: np.ndarray = field(default_factory=lambda: DEFAULT_INITIAL_STATE.copy())
    training_trajectories: int = 12
    validation_trajectories: int = 4
    trajectory_duration: float = 80.0
    input_window: int = 12
    glr_window: int = 20
    glr_min_segment: int = 3
    glr_threshold_quantile: float = 0.999
    covariance_regularization: float = 1e-6
    conv_channels: int = 16
    conv_kernel_size: int = 3
    hidden_size: int = 32
    lstm_layers: int = 1
    dropout: float = 0.0
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    @property
    def observation_dt(self) -> float:
        return self.dt * self.stride

    @property
    def checkpoint_path(self) -> Path:
        epoch = self.epochs if self.checkpoint_epoch is None else self.checkpoint_epoch
        return self.shared_config(checkpoint_epoch=epoch).checkpoint_path(epoch)

    def validate(self) -> None:
        resolve_noise_std(self.noise_std)
        initial = np.asarray(self.initial_state, dtype=float)
        if initial.shape != (STATE_DIM,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must contain three finite values")
        if self.t_end <= 0.0 or self.dt <= 0.0 or self.trajectory_duration <= 0.0:
            raise ValueError("time settings must be positive")
        if not 0.0 < self.switch_time < self.t_end:
            raise ValueError("switch_time must lie inside (0, t_end)")
        if not isinstance(self.stride, (int, np.integer)) or self.stride < 1:
            raise ValueError("stride must be a positive integer")
        if self.training_trajectories < 1 or self.validation_trajectories < 1:
            raise ValueError("trajectory counts must be positive")
        if self.input_window < 2:
            raise ValueError("input_window must be at least two")
        if self.glr_min_segment < 2 or self.glr_window < self.glr_min_segment:
            raise ValueError("GLR window settings are invalid")
        if not 0.0 < self.glr_threshold_quantile <= 1.0:
            raise ValueError("GLR threshold quantile must lie in (0, 1]")
        if self.covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        integer_settings = (
            self.conv_channels,
            self.conv_kernel_size,
            self.hidden_size,
            self.lstm_layers,
            self.epochs,
            self.batch_size,
        )
        if min(integer_settings) < 1:
            raise ValueError("network and training dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if self.checkpoint_epoch is not None and (
            self.checkpoint_epoch < 10 or self.checkpoint_epoch % 10
        ):
            raise ValueError('checkpoint_epoch must be None or a positive multiple of 10')

    def shared_config(self, checkpoint_epoch: int | None = None):
        return EXPERIMENT.ExperimentConfig(
            output_dir=Path(self.output_dir),
            checkpoint_epoch=checkpoint_epoch,
            seed=self.seed,
            t_end=self.t_end,
            switch_time=self.switch_time,
            dt=self.dt,
            stride=self.stride,
            state_dim=STATE_DIM,
            noise_std=resolve_noise_std(self.noise_std),
            training_trajectories=self.training_trajectories,
            validation_trajectories=self.validation_trajectories,
            trajectory_duration=self.trajectory_duration,
            input_window=self.input_window,
            glr_window=self.glr_window,
            glr_min_segment=self.glr_min_segment,
            glr_threshold_quantile=self.glr_threshold_quantile,
            covariance_regularization=self.covariance_regularization,
            conv_channels=self.conv_channels,
            conv_kernel_size=self.conv_kernel_size,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device="cpu",
        )


@dataclass(frozen=True)
class PredictorResult:
    model: nn.Module
    normalization: object
    validation_trajectories: list[np.ndarray]
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float | int]]
    source: str


def set_reproducible_seed(seed: int) -> None:
    EXPERIMENT.set_reproducible_seed(seed)


def resolve_device(requested: str = "cpu") -> torch.device:
    return EXPERIMENT.resolve_device(requested)


def build_predictor(config: DetectorConfig) -> nn.Module:
    config.validate()
    return EXPERIMENT.build_model(config.shared_config())


def _combine_windows(
    trajectories: list[np.ndarray],
    input_window: int,
    normalization: object,
) -> tuple[np.ndarray, np.ndarray]:
    return EXPERIMENT.combine_windows(trajectories, input_window, normalization)


def _validation_trajectories(config: DetectorConfig) -> list[np.ndarray]:
    return generate_normal_trajectories(
        config.validation_trajectories,
        config.trajectory_duration,
        config.dt,
        config.stride,
        config.noise_std,
        config.seed + 20_000,
    )


def _load_requested_checkpoint(config: DetectorConfig, device: torch.device) -> PredictorResult:
    if config.checkpoint_epoch is None:
        raise ValueError("an explicit checkpoint_epoch is required for loading")
    epoch = int(config.checkpoint_epoch)
    shared_config = config.shared_config(checkpoint_epoch=epoch)
    path = shared_config.checkpoint_path(epoch)
    loaded = EXPERIMENT.load_checkpoint(path, shared_config, device, epoch)
    return PredictorResult(
        model=loaded.model,
        normalization=loaded.normalization,
        validation_trajectories=_validation_trajectories(config),
        best_epoch=loaded.best_epoch,
        best_validation_loss=loaded.best_validation_loss,
        history=[],
        source="loaded",
    )


def _train_predictor_from_scratch(config: DetectorConfig, device: torch.device) -> PredictorResult:
    training = generate_normal_trajectories(
        config.training_trajectories,
        config.trajectory_duration,
        config.dt,
        config.stride,
        config.noise_std,
        config.seed + 10,
    )
    validation = _validation_trajectories(config)
    normalization = CORE.fit_normalization(training)
    train_inputs, train_targets = _combine_windows(training, config.input_window, normalization)
    validation_inputs, validation_targets = _combine_windows(
        validation, config.input_window, normalization
    )
    shared_config = config.shared_config()
    model = build_predictor(config).to(device)
    training_result = EXPERIMENT.train_model(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        normalization=normalization,
        config=shared_config,
        device=device,
    )
    best = EXPERIMENT.load_checkpoint(
        shared_config.checkpoint_path(training_result.best_epoch),
        shared_config,
        device,
        training_result.best_epoch,
    )
    history = training_result.history.to_dict(orient="records")
    return PredictorResult(
        model=best.model,
        normalization=best.normalization,
        validation_trajectories=validation,
        best_epoch=training_result.best_epoch,
        best_validation_loss=float(training_result.best_validation_loss),
        history=history,
        source="trained",
    )


def train_predictor(config: DetectorConfig, device: torch.device | None=None) -> PredictorResult:
    """Train by default, or load only an explicitly requested checkpoint."""
    config.validate()
    if device is None:
        device = torch.device("cpu")
    set_reproducible_seed(config.seed)
    if config.checkpoint_epoch is not None:
        return _load_requested_checkpoint(config, device)
    return _train_predictor_from_scratch(config, device)


def predict_trajectory(
    model: nn.Module,
    trajectory: np.ndarray,
    input_window: int,
    normalization: object,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return EXPERIMENT.predict_trajectory(
        model,
        trajectory,
        input_window,
        normalization,
        device,
        batch_size,
    )


def calibrate_predictor(predictor: PredictorResult, config: DetectorConfig, device: torch.device):
    residuals = [
        predict_trajectory(
            predictor.model,
            trajectory,
            config.input_window,
            predictor.normalization,
            device,
            config.batch_size,
        )[1]
        for trajectory in predictor.validation_trajectories
    ]
    return CORE.calibrate_glr(
        residuals,
        config.glr_window,
        config.glr_min_segment,
        config.glr_threshold_quantile,
        config.covariance_regularization,
    )


def evaluate_switched_trajectory(
    predictor: PredictorResult,
    calibration: object,
    config: DetectorConfig,
    device: torch.device,
) -> dict[str, object]:
    trajectory = simulate_sampled_trajectory(
        initial_state=config.initial_state,
        t_end=config.t_end,
        internal_dt=config.dt,
        observation_stride=config.stride,
        noise_std=config.noise_std,
        seed=config.seed + 30_000,
        switch_time=config.switch_time,
    )
    predictions, innovations = predict_trajectory(
        predictor.model,
        trajectory.sample_noisy,
        config.input_window,
        predictor.normalization,
        device,
        config.batch_size,
    )
    result = CORE.detect_switch(innovations, calibration)
    return {
        "trajectory": trajectory,
        "predictions": predictions,
        "innovations": innovations,
        "calibration": calibration,
        "result": result,
    }


def run_detection(config: DetectorConfig | None=None, device_name: str='cpu') -> dict[str, object]:
    if config is None:
        config = DetectorConfig()
    device = resolve_device(device_name)
    predictor = train_predictor(config, device=device)
    calibration = calibrate_predictor(predictor, config, device)
    run = evaluate_switched_trajectory(predictor, calibration, config, device)
    run.update(config=config, predictor=predictor, device=device)
    return run
