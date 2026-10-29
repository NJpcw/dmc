"""Evaluate one fixed predator-prey detector on 20 observation-noise seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.detection import residual_glr_detection as core, detector_training as training


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/predator_prey/detection_trials"


def load_or_train_without_csv(config, device):
    """Use the shared trainer without writing its optional history table."""
    return training.load_or_train_predictor(config, device, save_history=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=20)
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Load this saved checkpoint; omit to train once and select the best saved epoch.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory used only for detector checkpoint files.",
    )
    args = parser.parse_args()
    if args.seed_start < 1 or args.seed_end < args.seed_start:
        parser.error("seeds must form a positive, ordered range")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Keep this seed fixed: it determines the training and GLR calibration records.
    config = training.ExperimentConfig(
        output_dir=args.checkpoint_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        seed=2027,
        t_end=110.0,
        noise_std=0.1,
        glr_threshold_quantile=0.95,
    )
    config.validate()
    training.set_reproducible_seed(config.seed)
    device = training.resolve_device(config.device)
    predictor = load_or_train_without_csv(config, device)
    calibration = training.calibrate_predictor(predictor, config, device)

    training.evaluate_detection_trials(
        config, predictor, calibration, device,
        range(args.seed_start, args.seed_end + 1),
        np.array([config.initial_x1, config.initial_x2]),
        core.simulate_sampled_trajectory, training.predict_trajectory, core.detect_switch,
    )


if __name__ == "__main__":
    main()
