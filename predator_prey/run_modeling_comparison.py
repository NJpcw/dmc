"""Run the prespecified 20-seed four-method NGRC ablation comparison."""



from __future__ import annotations



import argparse

import json

import os

from pathlib import Path

import time



import numpy as np



os.environ.setdefault("MPLBACKEND", "Agg")



from common import prediction_metrics as reporting

from common.detection import residual_glr_detection as detection_core

from common.modeling_control import ngrc_modeling as modeling





REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/predator_prey/model_comparison"

FIXED_START_TIME = 100.4

TRAINING_DURATION = 2.0

METHOD_LABELS = reporting.METHOD_LABELS

build_seed_record = reporting.build_seed_record

summarize_seed_records = reporting.summarize_seed_records

plot_seed_comparison = reporting.plot_seed_comparison





def run_comparison(seeds=range(1, 21), output_dir=DEFAULT_OUTPUT):

    seeds = tuple(int(seed) for seed in seeds)

    if not seeds or any(seed <= 0 for seed in seeds):

        raise ValueError("At least one positive seed is required.")

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    started = time.perf_counter()

    clean_trajectory = detection_core.simulate_sampled_trajectory(

        initial_state=np.array([12.19, 24.4]),

        t_end=110.0,

        internal_dt=0.001,

        observation_stride=50,

        noise_std=0.0,

        seed=0,

        switch_time=100.0,

    )

    start_index = int(round(FIXED_START_TIME / 0.001))

    if not np.isclose(

        clean_trajectory.internal_time[start_index],

        FIXED_START_TIME,

        rtol=0.0,

        atol=1e-12,

    ):

        raise RuntimeError("The fixed modeling start time is off the integration grid.")

    fixed_start_state = clean_trajectory.internal_clean[start_index].copy()



    for position, seed in enumerate(seeds, start=1):

        config = modeling.ExperimentConfig(

            x0=fixed_start_state.copy(),

            t_end=9.0,

            noise_std=0.1,

            random_seed=seed,

            training_duration=TRAINING_DURATION,

            evaluation_duration=5.0,

            validation_horizon=20,

            external_horizon=20,

            external_anchors=60,

        )

        seed_started = time.perf_counter()

        results = modeling.run_experiment(

            config=config,

            methods=modeling.DEFAULT_COMPARISON_METHODS,

            make_plots=False,

            verbose=False,

        )

        record = build_seed_record(

            seed=seed,

            results=results,

            elapsed_seconds=time.perf_counter() - seed_started,

        )

        records.append(record)

        reporting.print_seed_metrics(position, len(seeds), record)



    aggregate = summarize_seed_records(records)

    summary = {

        "conditions": {

            "fixed_start_time": FIXED_START_TIME,

            "fixed_start_state": fixed_start_state.tolist(),

            "noise_std": 0.1,

            "training_duration_seconds": TRAINING_DURATION,

            "identification_dt": 0.01,

            "rollout_horizon": 20,

            "evaluation_duration_seconds": 5.0,

            "external_anchors_per_seed": 60,

            "seeds": list(seeds),

            "methods": METHOD_LABELS,

        },

        "aggregate": aggregate,

        "elapsed_seconds": float(time.perf_counter() - started),

        "seed_records": records,

    }

    plot_seed_comparison(records, output_dir / "seed_rmse_comparison.png")

    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)

    print(f"Artifacts: {output_dir}", flush=True)

    return summary





def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed-start", type=int, default=1)

    parser.add_argument("--seed-end", type=int, default=20)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    return parser.parse_args()





def main():

    arguments = parse_args()

    if arguments.seed_start <= 0 or arguments.seed_end < arguments.seed_start:

        raise ValueError("Seed range must be positive and ordered.")

    run_comparison(

        seeds=range(arguments.seed_start, arguments.seed_end + 1),

        output_dir=arguments.output_dir,

    )





if __name__ == "__main__":

    main()

