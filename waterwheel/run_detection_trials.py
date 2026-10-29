"""Evaluate one fixed waterwheel detector on 20 observation-noise seeds."""



from __future__ import annotations



import argparse

from pathlib import Path



from common.detection import residual_glr_detection as core, detector_training as training

from waterwheel import detection as detection, dynamics as system







DEFAULT_CHECKPOINT_DIR = detection.DetectorConfig().output_dir





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed-start", type=int, default=1)

    parser.add_argument("--seed-end", type=int, default=20)

    parser.add_argument(

        "--checkpoint-epoch",

        type=int,

        default=90,

        help="Load this saved checkpoint (default: epoch 90).",

    )

    parser.add_argument(

        "--checkpoint-dir",

        type=Path,

        default=DEFAULT_CHECKPOINT_DIR,

        help="Directory used only for detector checkpoint files.",

    )

    args = parser.parse_args()

    if args.seed_start < 1 or args.seed_end < args.seed_start:

        parser.error("seeds must form a positive, ordered range")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)



    # Keep this seed fixed: it determines training and GLR calibration records.

    config = detection.DetectorConfig(

        output_dir=args.checkpoint_dir,

        checkpoint_epoch=args.checkpoint_epoch,

    )

    config.validate()

    device = detection.resolve_device("cpu")

    predictor = detection.train_predictor(config, device=device)

    calibration = detection.calibrate_predictor(predictor, config, device)



    training.evaluate_detection_trials(

        config, predictor, calibration, device,

        range(args.seed_start, args.seed_end + 1), config.initial_state,

        system.simulate_sampled_trajectory, detection.predict_trajectory, core.detect_switch,

    )





if __name__ == "__main__":

    main()

