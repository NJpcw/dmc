"""Run the integrated multirate detection-modeling-control experiment."""

from __future__ import annotations

import argparse

from predator_prey.pipeline import (
    IntegratedPipelineConfig,
    run_integrated_pipeline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plots", action="store_true", help="Run without creating or showing plots."
    )
    arguments = parser.parse_args()
    config = IntegratedPipelineConfig()
    config.modeling_duration = 2.0
    config.model_config.training_duration = 2.0
    run_integrated_pipeline(config=config, make_plots=not arguments.no_plots, verbose=True)


if __name__ == "__main__":
    main()
