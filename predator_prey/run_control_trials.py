"""Run fixed-model predator-prey control comparisons on multiple noise seeds."""



from __future__ import annotations



import argparse

from dataclasses import replace

from types import SimpleNamespace



import numpy as np



from common.control_experiment_utils import (
    _nan_metrics,
    _process_index,
    APPROVED_METRICS,
    extract_control_metrics,
    generate_measurement_noise,
    generate_ou_disturbance,
    print_aggregate_metrics,
    print_trial_metrics,
    spawn_trial_generators,
)

from common.detection import residual_glr_detection as detection_core

from common.modeling_control import ngrc_modeling as modeling

from common.post_detection_data import prepare_modeling_segment

from predator_prey import pipeline as pipeline





FIXED_START_TIME = 100.4

MODELING_DURATION = 2.0

MODELING_SEED = 40_001

MEASUREMENT_NOISE_STD = np.array([0.1, 0.1], dtype=float)

PROCESS_NOISE_STD = np.array([0.2, 0.2], dtype=float)

PROCESS_CORRELATION_TIME = 0.1



CONTROL_METHODS = {

    "Raw-NGRC+MPC": ("raw", "mpc"),

    "Raw-delay-NGRC+MPC": ("raw_delay", "mpc"),

    "BLI-NGRC+MPC": ("bli", "mpc"),

    "BLI-NGRC+ESO-ADR+MPC": ("bli", "eso_adr_mpc"),

    "Raw-NGRC+ESO-ADR+MPC": ("raw", "eso_adr_mpc"),

}





def _fixed_alarm_observation(clean_state: np.ndarray) -> np.ndarray:

    rng = np.random.default_rng(1_000_000 + MODELING_SEED)

    return np.asarray(clean_state, dtype=float) + rng.normal(0.0, MEASUREMENT_NOISE_STD)





def prepare_fixed_models():

    """Fit the three control models once at the prescribed fixed start time."""

    config = pipeline.IntegratedPipelineConfig()

    config.modeling_duration = MODELING_DURATION

    config.modeling_noise_std = float(MEASUREMENT_NOISE_STD[0])

    config.modeling_seed = MODELING_SEED

    config.model_config = replace(config.model_config, training_duration=MODELING_DURATION)

    config.validate()



    trajectory = detection_core.simulate_sampled_trajectory(

        initial_state=np.array([12.19, 24.4], dtype=float),

        t_end=FIXED_START_TIME + MODELING_DURATION + 0.1,

        internal_dt=0.001,

        observation_stride=50,

        noise_std=0.0,

        seed=0,

        switch_time=100.0,

    )

    start_sample_index = int(np.searchsorted(trajectory.sample_time, FIXED_START_TIME, side='left'))

    if start_sample_index >= len(trajectory.sample_time) or not np.isclose(

        trajectory.sample_time[start_sample_index],

        FIXED_START_TIME,

        rtol=0.0,

        atol=1e-12,

    ):

        raise RuntimeError("The fixed modeling start is off the observation grid")

    sample_noisy = np.asarray(trajectory.sample_clean, dtype=float).copy()

    sample_noisy[start_sample_index] = _fixed_alarm_observation(

        trajectory.sample_clean[start_sample_index]

    )

    trajectory = replace(trajectory, sample_noisy=sample_noisy)

    fixed_start_result = SimpleNamespace(alarm_index=start_sample_index)

    segment = prepare_modeling_segment(

        trajectory=trajectory,

        detection_result=fixed_start_result,

        duration=MODELING_DURATION,

        modeling_stride=config.modeling_stride,

        noise_std=config.modeling_noise_std,

        seed=MODELING_SEED,

    )

    models = {

        method: modeling.train_method_from_observations(

            segment.sample_time,

            segment.sample_noisy,

            config.model_config,

            method=method,

        )

        for method in ("raw", "raw_delay", "bli")

    }

    return config, segment, models





def make_trial_plant(process_path: np.ndarray, handoff_time: float, control_config):

    """Create a stateless plant step backed by one immutable OU path."""

    process_path = np.asarray(process_path, dtype=float)

    if process_path.ndim != 2 or process_path.shape[1] != 2:

        raise ValueError("Predator-prey process_path must have shape [time, 2]")

    process_path.setflags(write=False)



    def plant_step(state, time, control_input, internal_dt, substeps, with_disturbance):

        if not np.isclose(internal_dt, control_config.internal_dt, rtol=0.0, atol=1e-15):

            raise ValueError("Plant and configured internal time steps must agree")

        next_state = np.asarray(state, dtype=float).copy()

        next_time = float(time)

        for _ in range(int(substeps)):

            rate = pipeline.CONTROL.controlled_grassland_rhs(

                next_state,

                next_time,

                control_input,

                with_disturbance=with_disturbance,

            )

            if with_disturbance:

                index = _process_index(next_time, handoff_time, internal_dt, len(process_path))

                rate = rate + process_path[index]

            next_state = np.maximum(next_state + internal_dt * rate, 1e-8)

            next_time += internal_dt

        return next_state



    return plant_step





def run_one_trial(seed, config, segment, models):

    control_config = config.control_config

    control_dt = float(models["bli"]["identification_dt"])

    num_control_steps = int(round(control_config.control_duration / control_dt))

    measurement_rng, process_rng = spawn_trial_generators(int(seed))

    measurement_noise = generate_measurement_noise(

        measurement_rng,

        sample_count=num_control_steps + 1,

        noise_std=MEASUREMENT_NOISE_STD,

    )

    process_path = generate_ou_disturbance(

        process_rng,

        sample_count=num_control_steps * control_config.control_stride + 1,

        dt=control_config.internal_dt,

        tau=PROCESS_CORRELATION_TIME,

        stationary_std=PROCESS_NOISE_STD,

    )

    handoff_true_state = np.asarray(segment.sample_clean[-1], dtype=float)

    handoff_measurement = handoff_true_state + measurement_noise[0]

    future_measurement_noise = measurement_noise[1:]

    handoff_time = float(segment.sample_time[-1])

    method_metrics = {}

    for label, (model_key, controller_type) in CONTROL_METHODS.items():

        try:

            simulation = pipeline.run_controller_from_modeling_handoff(

                model=models[model_key],

                modeling_segment=segment,

                control_config=control_config,

                controller_type=controller_type,

                future_measurement_noise=future_measurement_noise,

                handoff_measurement=handoff_measurement,

                plant_step=make_trial_plant(process_path, handoff_time, control_config),

            )

            method_metrics[label] = extract_control_metrics(

                simulation,

                window_duration=min(2.0, control_config.control_duration),

            )

        except Exception as error:

            print(f"  seed={seed:02d} {label} failed: {error}", flush=True)

            method_metrics[label] = _nan_metrics()

    return method_metrics





def run_trials(seeds=range(1, 21)):

    seeds = tuple(int(seed) for seed in seeds)

    if not seeds or any(seed <= 0 for seed in seeds):

        raise ValueError("At least one positive seed is required")

    config, segment, models = prepare_fixed_models()

    print("Predator-prey fixed-model control trials", flush=True)

    print(

        f"modeling_start={FIXED_START_TIME} "

        f"modeling_duration={MODELING_DURATION} "

        f"modeling_seed={MODELING_SEED}",

        flush=True,

    )

    print(

        f"measurement_noise_std={MEASUREMENT_NOISE_STD.tolist()} "

        f"process_noise_std={PROCESS_NOISE_STD.tolist()} "

        f"process_tau={PROCESS_CORRELATION_TIME}",

        flush=True,

    )

    print(

        "selected_alpha="

        + ", ".join(

            f"{method}:{models[method]['selected_alpha']:.8g}"

            for method in ("raw", "raw_delay", "bli")

        ),

        flush=True,

    )

    records = []

    method_order = tuple(CONTROL_METHODS)

    for position, seed in enumerate(seeds, start=1):

        method_metrics = run_one_trial(seed, config, segment, models)

        records.append(method_metrics)

        print_trial_metrics(position, len(seeds), seed, method_metrics, method_order)

    print_aggregate_metrics(records, method_order)

    return records





def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed-start", type=int, default=1)

    parser.add_argument("--seed-end", type=int, default=20)

    return parser.parse_args()





def main():

    arguments = parse_args()

    if arguments.seed_start <= 0 or arguments.seed_end < arguments.seed_start:

        raise ValueError("Seed range must be positive and ordered")

    run_trials(range(arguments.seed_start, arguments.seed_end + 1))





if __name__ == "__main__":

    main()

