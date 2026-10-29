"""Waterwheel adapter for the shared NGRC--ESO--CasADi MPC engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.modeling_control import disturbance_rejection_mpc as SHARED_CONTROL

from . import modeling as modeling
from .dynamics import DEFAULT_NOISE_STD, Q_AFTER, STATE_DIM, waterwheel_rhs


REFERENCE_OMEGA = float(np.sqrt(17.0))
REFERENCE_B = Q_AFTER / (REFERENCE_OMEGA**2 + 1.0)
REFERENCE_A = REFERENCE_OMEGA * REFERENCE_B
NORMAL_REFERENCE = np.array([REFERENCE_OMEGA, REFERENCE_A, REFERENCE_B], dtype=float)
STEADY_CONTROL = 3.0 * (REFERENCE_OMEGA - REFERENCE_A)
INPUT_VECTOR = np.array([1.0, 0.0, 0.0], dtype=float)
INPUT_MATRIX = INPUT_VECTOR[:, None]

NGRCResidualESO = SHARED_CONTROL.NGRCResidualESO
solve_warmstart_mpc = SHARED_CONTROL.solve_warmstart_mpc


@dataclass
class ControlConfig(SHARED_CONTROL.ControlConfig):
    """Waterwheel defaults supplied to the shared controller configuration."""

    model_key: str = "bli"
    reference: np.ndarray = field(default_factory=lambda: NORMAL_REFERENCE.copy())
    control_duration: float = 6.0
    horizon: int = 20
    move_horizon: int | None = 10
    internal_dt: float = 0.001
    control_stride: int = 10
    input_matrix: np.ndarray = field(default_factory=lambda: INPUT_MATRIX.copy())
    state_lower_bound: np.ndarray | float | None = None
    measurement_noise_std: np.ndarray = field(default_factory=lambda: DEFAULT_NOISE_STD.copy())
    observer_bandwidth: float = 10.0
    q_state: np.ndarray = field(default_factory=lambda: np.array([30.0, 20.0, 10.0], dtype=float))
    r_control: float = 0.02
    r_delta: float = 0.12
    terminal_weight: float = 5.0
    u_min: np.ndarray = field(default_factory=lambda: np.array([-4.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([4.0]))
    du_max: np.ndarray = field(default_factory=lambda: np.array([0.25]))
    mpc_maxiter: int = 60
    mpc_ftol: float = 1e-5
    disturbance_scale: float = 1.0

    @property
    def control_dt(self) -> float:
        return self.internal_dt * self.control_stride

    def validate(self, model_dt: float | None = None) -> None:
        dt = self.control_dt if model_dt is None else float(model_dt)
        super().validate(dt)
        if not np.isfinite(self.disturbance_scale) or self.disturbance_scale < 0.0:
            raise ValueError("disturbance_scale must be finite and non-negative")


def unknown_total_disturbance(state: np.ndarray, time: float) -> float:
    state = np.asarray(state, dtype=float)
    return 2 * float(
        # 0.8
        + 0.18 * np.sin(0.45 * time)
        + 0.08 * np.cos(0.17 * time)
        + 0.005 * (state[0] - REFERENCE_OMEGA)
    )


def controlled_waterwheel_rhs(
    state: np.ndarray,
    time: float,
    control: float,
    with_disturbance: bool = True,
    disturbance_scale: float = 1.0,
) -> np.ndarray:
    disturbance = (
        disturbance_scale * unknown_total_disturbance(state, time) if with_disturbance else 0.0
    )
    return waterwheel_rhs(
        state,
        q=Q_AFTER,
        control=float(np.asarray(control, dtype=float)),
        disturbance=disturbance,
    )


def euler_step_controlled_plant(
    state: np.ndarray,
    time: float,
    control: float | np.ndarray,
    internal_dt: float,
    substeps: int,
    with_disturbance: bool,
    disturbance_scale: float = 1.0,
) -> np.ndarray:
    if internal_dt <= 0.0 or substeps < 1:
        raise ValueError("internal_dt and substeps must be positive")
    control_vector = np.asarray(control, dtype=float).reshape(-1)
    if control_vector.shape != (1,):
        raise ValueError("the waterwheel plant requires exactly one control input")
    next_state = np.asarray(state, dtype=float).copy()
    next_time = float(time)
    for _ in range(int(substeps)):
        next_state = next_state + internal_dt * controlled_waterwheel_rhs(
            next_state,
            next_time,
            float(control_vector[0]),
            with_disturbance=with_disturbance,
            disturbance_scale=disturbance_scale,
        )
        next_time += internal_dt
    return next_state


def simulate_controller_from_handoff(
    model: dict[str, object],
    modeling_segment,
    config: ControlConfig,
    controller_type: str = "mpc",
    future_measurement_noise: np.ndarray | None = None,
    handoff_measurement: np.ndarray | None = None,
    plant_step=None,
) -> dict[str, object]:
    """Adapt the waterwheel handoff to the shared closed-loop simulator."""
    shared_controller_type = {
        "mpc": "mpc",
        "adr_mpc": "eso_adr_mpc",
        "eso_adr_mpc": "eso_adr_mpc",
    }.get(controller_type)
    if shared_controller_type is None:
        raise ValueError("controller_type must be 'mpc', 'adr_mpc', or 'eso_adr_mpc'")
    model_dt = float(model.get("identification_dt", config.control_dt))
    config.validate(model_dt)
    required = modeling.minimum_history_length(model)
    model_source = np.asarray(model.get('X_input', modeling_segment.sample_noisy), dtype=float)
    true_source = np.asarray(modeling_segment.sample_clean, dtype=float)
    if len(model_source) < required or len(true_source) < required:
        raise ValueError("the fitted model does not contain enough handoff history")

    def default_plant_step(state, time, control_input, **settings):
        return euler_step_controlled_plant(
            state,
            time,
            control_input,
            disturbance_scale=config.disturbance_scale,
            **settings,
        )

    if handoff_measurement is None:
        handoff_measurement = np.asarray(modeling_segment.sample_noisy[-1], dtype=float)
    selected_plant_step = default_plant_step if plant_step is None else plant_step

    result = SHARED_CONTROL.simulate_controller_from_handoff(
        model=model,
        config=config,
        controller_type=shared_controller_type,
        handoff_time=float(modeling_segment.sample_time[-1]),
        handoff_true_state=np.asarray(modeling_segment.sample_clean[-1], dtype=float),
        handoff_measurement=handoff_measurement,
        model_history=model_source[-required:],
        true_history=true_source[-required:],
        future_measurement_noise=future_measurement_noise,
        plant_step=selected_plant_step,
    )
    result.update(
        controller_type=controller_type,
        U=result["U_actual"][:, 0],
        disturbance_estimate=result["D_hat"][:, 0],
        solver_success=result["optimizer_success"],
        reference=result["x_ref"],
        measurement_noise=(
            np.asarray(future_measurement_noise, dtype=float).copy()
            if future_measurement_noise is not None
            else result["X_meas"][1:] - result["X_true"][1:]
        ),
    )
    return result


def tracking_rmse(result: dict[str, object]) -> float:
    return SHARED_CONTROL.tracking_rmse(result)


def steady_state_metrics(
    result: dict[str, object], window_duration: float = 2.0
) -> dict[str, object]:
    mask, final_duration = SHARED_CONTROL.final_window_mask(result, window_duration)
    states = np.asarray(result["X_true"], dtype=float)
    reference = np.asarray(
        result["x_ref"] if "x_ref" in result else result["reference"],
        dtype=float,
    )
    if states.shape != (len(np.asarray(result["t"])), STATE_DIM):
        raise ValueError("time and three-state trajectory lengths must agree")
    if reference.shape != (STATE_DIM,):
        raise ValueError("reference must contain three states")
    error = states[mask] - reference
    return {
        "steady_state_window": final_duration,
        "steady_state_rmse": float(np.sqrt(np.mean(error**2))),
        "per_state_mae": np.mean(np.abs(error), axis=0).tolist(),
        "final_abs_error": np.abs(states[-1] - reference).tolist(),
    }


def control_energy(result: dict[str, object]) -> float:
    return SHARED_CONTROL.control_energy(result)


def control_metric_panel(
    result: dict[str, object], window_duration: float = 2.0
) -> dict[str, object]:
    return SHARED_CONTROL.control_metric_panel(result, window_duration)


def constraint_violations(result: dict[str, object], config: ControlConfig) -> dict[str, int]:
    controls = np.asarray(
        result["U_actual"] if "U_actual" in result else result["U"], dtype=float
    ).reshape(-1, 1)
    lower = np.asarray(config.u_min, dtype=float)
    upper = np.asarray(config.u_max, dtype=float)
    slew = np.asarray(config.du_max, dtype=float)
    input_violations = (controls < lower - 1e-9) | (controls > upper + 1e-9)
    increments = np.diff(np.vstack([np.zeros((1, 1)), controls]), axis=0)
    slew_violations = np.abs(increments) > slew + 1e-9
    return {
        "input_violation_count": int(np.count_nonzero(input_violations)),
        "slew_violation_count": int(np.count_nonzero(slew_violations)),
    }
