"""End-to-end multirate detection, NGRC modeling, and MPC control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.detection import residual_glr_detection as DETECTOR_CORE
from common.detection import detector_training as DETECTOR_EXPERIMENT
from common.modeling_control import ngrc_modeling as NGRC
from common.modeling_control import disturbance_rejection_mpc as CONTROL
from common.post_detection_data import (
    build_rolling_prediction_results,
    prepare_modeling_segment,
    save_matlab_plot_data,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _default_detector_config():
    return DETECTOR_EXPERIMENT.ExperimentConfig(
        output_dir=REPOSITORY_ROOT / "outputs/predator_prey/pipeline",
        checkpoint_epoch=100,
        seed=2027,
        t_end=200.0,
        switch_time=100.0,
        dt=0.001,
        stride=50,
        noise_std=0.1,
        initial_x1=12.19,
        initial_x2=24.4,
        training_trajectories=12,
        validation_trajectories=4,
        trajectory_duration=80.0,
        input_window=12,
        glr_window=20,
        glr_min_segment=3,
        glr_threshold_quantile=0.95,
        covariance_regularization=1e-6,
        conv_channels=16,
        conv_kernel_size=3,
        hidden_size=32,
        lstm_layers=2,
        dropout=0.0,
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cpu",
    )


@dataclass
class IntegratedPipelineConfig:
    detector_config: object = field(default_factory=_default_detector_config)
    model_config: object = field(default_factory=lambda: NGRC.ExperimentConfig(noise_std=0.1))
    control_config: object = field(default_factory=lambda: CONTROL.ControlConfig(measurement_noise_std=0.1))
    modeling_duration: float = 3.0
    modeling_stride: int = 10
    modeling_noise_std: float = 0.1
    modeling_seed: int = 40_001

    def validate(self):
        self.detector_config.validate()
        self.model_config.validate()
        model_dt = float(self.model_config.identification_dt)
        self.control_config.validate(model_dt)
        if not np.isclose(self.detector_config.dt, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("detector internal dt must be 0.001 s")
        if self.detector_config.stride != 50:
            raise ValueError("detection stride must be 50")
        if not np.isclose(self.model_config.dt_original, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("model internal dt must be 0.001 s")
        if self.model_config.stride != self.modeling_stride or self.modeling_stride != 10:
            raise ValueError("modeling stride must be 10")
        if not np.isclose(self.control_config.internal_dt, 0.001, rtol=0.0, atol=1e-12):
            raise ValueError("control internal dt must be 0.001 s")
        if self.control_config.control_stride != 10:
            raise ValueError("control stride must be 10")
        if self.control_config.model_key != "bli":
            raise ValueError("integrated control model_key must be 'bli'")
        if not np.isclose(
            self.modeling_duration,
            self.model_config.training_duration,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("modeling duration must equal NGRC training duration")
        if self.modeling_noise_std < 0:
            raise ValueError("modeling noise standard deviation cannot be negative")


def _shared_future_measurement_noise(model, control_config):
    model_dt = float(
        model.get('identification_dt', control_config.internal_dt * control_config.control_stride)
    )
    control_config.validate(model_dt)
    num_steps = int(round(control_config.control_duration / model_dt))
    rng = np.random.default_rng(control_config.random_seed)
    return control_config.measurement_noise_std * rng.standard_normal(
        (num_steps, len(np.asarray(control_config.reference)))
    )


def run_controller_from_modeling_handoff(
    model,
    modeling_segment,
    control_config,
    controller_type,
    future_measurement_noise,
    handoff_measurement=None,
    plant_step=None,
):
    required = CONTROL.minimum_history_length(model)
    if len(model["X_input"]) < required or len(modeling_segment.sample_clean) < required:
        raise ValueError("Modeling histories are too short for the controller handoff.")
    model_dt = float(
        model.get('identification_dt', control_config.internal_dt * control_config.control_stride)
    )
    control_config.validate(model_dt)
    if handoff_measurement is None:
        handoff_measurement = np.asarray(modeling_segment.sample_noisy[-1], dtype=float)
    common = dict(
        model=model,
        config=control_config,
        handoff_time=float(modeling_segment.sample_time[-1]),
        handoff_true_state=np.asarray(modeling_segment.sample_clean[-1], dtype=float),
        handoff_measurement=handoff_measurement,
        model_history=np.asarray(model["X_input"][-required:], dtype=float),
        true_history=np.asarray(modeling_segment.sample_clean[-required:], dtype=float),
        future_measurement_noise=future_measurement_noise,
    )
    if plant_step is not None:
        common["plant_step"] = plant_step
    return CONTROL.simulate_controller_from_handoff(controller_type=controller_type, **common)


def run_control_from_modeling_handoff(model, modeling_segment, control_config):
    shared_future_noise = _shared_future_measurement_noise(model, control_config)
    ordinary = run_controller_from_modeling_handoff(
        model,
        modeling_segment,
        control_config,
        controller_type="mpc",
        future_measurement_noise=shared_future_noise,
    )
    eso = run_controller_from_modeling_handoff(
        model,
        modeling_segment,
        control_config,
        controller_type="eso_adr_mpc",
        future_measurement_noise=shared_future_noise,
    )
    return ordinary, eso


def run_control_benchmarks_from_modeling_handoff(
    raw_model,
    raw_delay_model,
    bli_model,
    modeling_segment,
    control_config,
):
    raw_dt = float(raw_model["identification_dt"])
    raw_delay_dt = float(raw_delay_model["identification_dt"])
    bli_dt = float(bli_model["identification_dt"])
    if not (
        np.isclose(raw_dt, raw_delay_dt, rtol=0.0, atol=1e-12)
        and np.isclose(raw_dt, bli_dt, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("All NGRC control models must use the same sampling interval.")
    shared_future_noise = _shared_future_measurement_noise(bli_model, control_config)
    raw_mpc = run_controller_from_modeling_handoff(
        raw_model,
        modeling_segment,
        control_config,
        controller_type="mpc",
        future_measurement_noise=shared_future_noise,
    )
    raw_delay_mpc = run_controller_from_modeling_handoff(
        raw_delay_model,
        modeling_segment,
        control_config,
        controller_type="mpc",
        future_measurement_noise=shared_future_noise,
    )
    bli_mpc = run_controller_from_modeling_handoff(
        bli_model,
        modeling_segment,
        control_config,
        controller_type="mpc",
        future_measurement_noise=shared_future_noise,
    )
    bli_adr = run_controller_from_modeling_handoff(
        bli_model,
        modeling_segment,
        control_config,
        controller_type="eso_adr_mpc",
        future_measurement_noise=shared_future_noise,
    )
    raw_adr = run_controller_from_modeling_handoff(
        raw_model,
        modeling_segment,
        control_config,
        controller_type="eso_adr_mpc",
        future_measurement_noise=shared_future_noise,
    )
    return raw_mpc, raw_delay_mpc, bli_mpc, bli_adr, raw_adr


def print_integrated_summary(results):
    detection = results["detection"]
    modeling = results["modeling"]
    models = results["models"]
    print("========== Integrated multirate pipeline ==========")
    print(
        f"true switch = {detection['true_switch_time']:.3f} | "
        f"estimated = {detection['estimated_switch_time']:.3f} | "
        f"formal alarm = {detection['alarm_time']:.3f}"
    )
    print(
        f"alarm delay = {detection['alarm_delay']:.6f} | "
        f"localization error = {detection['localization_error']:.6f} | "
        f"false alarm = {detection['false_alarm']}"
    )
    print(
        f"modeling = [{modeling.sample_time[0]:.3f}, "
        f"{modeling.sample_time[-1]:.3f}] | samples = {len(modeling.sample_time)}"
    )
    print(
        f"raw alpha = {models['raw']['selected_alpha']:.8g} | "
        f"raw-delay alpha = {models['raw_delay']['selected_alpha']:.8g} | "
        f"bli alpha = {models['bli']['selected_alpha']:.8g}"
    )
    NGRC.print_experiment_summary(results["rolling_predictions"])
    CONTROL.print_control_summary(
        results["ordinary_control"],
        results["eso_control"],
        final_window_duration=results["config"].control_config.steady_state_window,
        raw_mpc=results["raw_mpc_control"],
        raw_delay_mpc=results["raw_delay_mpc_control"],
        raw_adr_mpc=results["raw_adr_control"],
    )


def run_integrated_pipeline(
    config=None, make_plots=True, verbose=True, *, matlab_output_dir=None
):
    if config is None:
        config = IntegratedPipelineConfig()
    config.validate()
    detector_config = config.detector_config
    DETECTOR_EXPERIMENT.set_reproducible_seed(detector_config.seed)
    device = DETECTOR_EXPERIMENT.resolve_device(detector_config.device)
    predictor = DETECTOR_EXPERIMENT.load_or_train_predictor(detector_config, device)
    calibration = DETECTOR_EXPERIMENT.calibrate_predictor(predictor, detector_config, device)

    trajectory = DETECTOR_CORE.simulate_sampled_trajectory(
        initial_state=np.array([detector_config.initial_x1, detector_config.initial_x2]),
        t_end=detector_config.t_end,
        internal_dt=detector_config.dt,
        observation_stride=detector_config.stride,
        noise_std=detector_config.noise_std,
        seed=detector_config.seed + 30_000,
        # seed=32031,
        switch_time=detector_config.switch_time,
    )
    predictions, innovations = DETECTOR_EXPERIMENT.predict_trajectory(
        predictor.model,
        trajectory.sample_noisy,
        detector_config.input_window,
        predictor.normalization,
        device,
        detector_config.batch_size,
    )
    detection_result = DETECTOR_CORE.detect_switch(innovations, calibration)
    if detection_result.alarm_index is None:
        raise RuntimeError("No formal GLR alarm occurred; NGRC modeling cannot start.")

    alarm_index = int(detection_result.alarm_index)
    estimated_index = int(detection_result.estimated_switch_index)
    modeling_segment = prepare_modeling_segment(
        trajectory=trajectory,
        detection_result=detection_result,
        duration=config.modeling_duration,
        modeling_stride=config.modeling_stride,
        noise_std=config.modeling_noise_std,
        seed=config.modeling_seed,
    )
    raw_model = NGRC.train_method_from_observations(
        modeling_segment.sample_time,
        modeling_segment.sample_noisy,
        config.model_config,
        method="raw",
    )
    raw_delay_model = NGRC.train_method_from_observations(
        modeling_segment.sample_time,
        modeling_segment.sample_noisy,
        config.model_config,
        method="raw_delay",
    )
    bli_model = NGRC.train_method_from_observations(
        modeling_segment.sample_time,
        modeling_segment.sample_noisy,
        config.model_config,
        method="bli",
    )
    models = {"raw": raw_model, "raw_delay": raw_delay_model, "bli": bli_model}
    rolling_predictions = build_rolling_prediction_results(
        trajectory=trajectory,
        detection_result=detection_result,
        models=models,
        config=config,
    )
    raw_mpc, raw_delay_mpc, bli_mpc, bli_adr, raw_adr = (
        run_control_benchmarks_from_modeling_handoff(
            raw_model,
            raw_delay_model,
            bli_model,
            modeling_segment,
            config.control_config,
        )
    )
    detection = {
        "trajectory": trajectory,
        "predictions": predictions,
        "innovations": innovations,
        "result": detection_result,
        "calibration": calibration,
        "true_switch_time": float(detector_config.switch_time),
        "estimated_switch_time": float(trajectory.sample_time[estimated_index]),
        "alarm_time": float(trajectory.sample_time[alarm_index]),
        "alarm_delay": float(trajectory.sample_time[alarm_index] - detector_config.switch_time),
        "localization_error": float(abs(
            trajectory.sample_time[estimated_index] - detector_config.switch_time
        )),
        "false_alarm": bool(trajectory.sample_time[alarm_index] < detector_config.switch_time),
    }
    results = {
        "config": config,
        "predictor": predictor,
        "detection": detection,
        "modeling": modeling_segment,
        "model": bli_model,
        "models": models,
        "rolling_predictions": rolling_predictions,
        "raw_mpc_control": raw_mpc,
        "raw_delay_mpc_control": raw_delay_mpc,
        "bli_mpc_control": bli_mpc,
        "bli_adr_control": bli_adr,
        "raw_adr_control": raw_adr,
        "ordinary_control": bli_mpc,
        "eso_control": bli_adr,
    }
    results["matlab_data_path"] = save_matlab_plot_data(
        results, matlab_output_dir if matlab_output_dir is not None else detector_config.output_dir
    )
    if verbose:
        print_integrated_summary(results)
        print(f"MATLAB plot data: {results['matlab_data_path']}")
    if make_plots:
        output_dir = Path(detector_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        DETECTOR_EXPERIMENT.plot_experiment(
            output_dir / "integrated_detection.png",
            trajectory.sample_time,
            trajectory.sample_clean,
            trajectory.sample_noisy,
            innovations,
            detection_result,
            calibration,
            detector_config.switch_time,
            detection["alarm_time"],
            detection["estimated_switch_time"],
        )
        NGRC.plot_rollout_examples(
            rolling_predictions,
            num_anchors=6,
            output_path=output_dir / "integrated_ngrc_rolling_prediction.png",
        )
        CONTROL.plot_control_comparison(
            bli_mpc,
            bli_adr,
            raw_mpc=raw_mpc,
            raw_delay_mpc=raw_delay_mpc,
            raw_adr_mpc=raw_adr,
            output_path=output_dir / "integrated_control_comparison.png",
            show=False,
        )
    return results
