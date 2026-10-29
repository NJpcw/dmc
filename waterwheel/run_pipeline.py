"""Command-line entry point for the waterwheel integrated pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from waterwheel.pipeline import PipelineConfig, make_quick_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the reduced-cost integration configuration.",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=90,
        help="Load this compatible detector checkpoint; default trains from scratch.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write figures or MATLAB plot data.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()

    if arguments.quick:
        config = make_quick_config(arguments.output_dir)
    else:
        config = PipelineConfig()
        config.modeling_duration = 3.0
        config.model_config = replace(config.model_config, training_duration=3.0)
        if arguments.output_dir is not None:
            config.output_dir = arguments.output_dir
            config.detector_config = replace(
                config.detector_config,
                output_dir=arguments.output_dir / "detection",
            )
    if arguments.checkpoint_epoch is not None:
        config.detector_config = replace(
            config.detector_config,
            checkpoint_epoch=arguments.checkpoint_epoch,
        )
    run_pipeline(config, save_results=not arguments.no_save, verbose=True)


if __name__ == "__main__":
    main()
