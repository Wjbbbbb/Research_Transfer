"""Leakage-safe group splits for M1 curriculum experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPLIT_NAMES = ("train", "calibration", "test")


def mindcube_group_key(sample: dict[str, Any]) -> str:
    """Return the shared frame-directory identity for one MindCube question."""

    images = sample.get("images") or sample.get("image")
    if not isinstance(images, list) or not images or not all(isinstance(path, str) for path in images):
        raise ValueError(f"MindCube sample {sample.get('id', '<unknown>')} has no valid image list")
    parents = {str(Path(path).parent) for path in images}
    if len(parents) != 1:
        raise ValueError(f"MindCube sample {sample.get('id', '<unknown>')} spans multiple frame groups")
    return parents.pop()


def assign_group(group_key: str, *, seed: int, train_fraction: float, calibration_fraction: float) -> str:
    """Assign an entire frame group deterministically, independent of input order."""

    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction + calibration_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test split")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + calibration_fraction:
        return "calibration"
    return "test"


def split_mindcube_samples(
    samples: Iterable[dict[str, Any]],
    *,
    seed: int = 0,
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.1,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Split samples by frame group and return both samples and group assignment."""

    assignments: dict[str, str] = {}
    split_samples = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        group = mindcube_group_key(sample)
        split = assignments.setdefault(
            group,
            assign_group(group, seed=seed, train_fraction=train_fraction, calibration_fraction=calibration_fraction),
        )
        split_samples[split].append(sample)
    if not assignments:
        raise ValueError("cannot split an empty MindCube sample list")
    return split_samples, assignments


def write_mindcube_splits(
    source: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.1,
) -> dict[str, Any]:
    """Materialize immutable split files plus a provenance manifest.

    The destination must not already contain any planned file, preventing an
    accidental replacement of a split that underlies an ongoing experiment.
    """

    source_path, destination = Path(source), Path(output_dir)
    samples = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("MindCube annotation source must contain a JSON list")
    split_samples, assignments = split_mindcube_samples(
        samples,
        seed=seed,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )
    planned = [destination / f"{name}.json" for name in SPLIT_NAMES] + [destination / "manifest.json"]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing split artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        (destination / f"{name}.json").write_text(json.dumps(split_samples[name], ensure_ascii=False) + "\n", encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "dualgeo.m1.mindcube-split.v1",
        "source": str(source_path),
        "source_sha256": source_hash,
        "group_key": "shared parent directory of every image path in a sample",
        "seed": seed,
        "fractions": {"train": train_fraction, "calibration": calibration_fraction, "test": 1 - train_fraction - calibration_fraction},
        "samples": {name: len(split_samples[name]) for name in SPLIT_NAMES},
        "groups": dict(Counter(assignments.values())),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
