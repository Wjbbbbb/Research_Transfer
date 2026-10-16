"""Create leakage-safe MindCube M1 train/calibration/test split artifacts."""

from __future__ import annotations

import argparse
import json

from .splits import write_mindcube_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = write_mindcube_splits(
        args.source,
        args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
