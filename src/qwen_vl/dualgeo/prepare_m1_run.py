"""Create the immutable, local manifest for one M1 curriculum run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def write_manifest(
    run_root: Path,
    *,
    order: str,
    steps: int,
    total_batch_size: int,
    world_size: int,
    qwen_checkpoint: Path,
    vggt_checkpoint: Path,
    sidecar_checkpoint: Path,
    split_root: Path,
    repo_root: Path,
) -> Path:
    """Write once; refusing overwrite prevents accidental run conflation."""

    destination = run_root / "run_manifest.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing M1 manifest: {destination}")
    source_manifest = split_root / "manifest.json"
    payload = {
        "schema_version": "dualgeo.m1.curriculum-run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_revision(repo_root),
        "order": order,
        "stage_max_steps": steps,
        "total_batch_size": total_batch_size,
        "world_size": world_size,
        "seed": 0,
        "qwen_checkpoint": str(qwen_checkpoint),
        "vggt_checkpoint": str(vggt_checkpoint),
        "seek_sidecar_checkpoint": str(sidecar_checkpoint),
        "split_manifest": str(source_manifest),
        "split_manifest_sha256": sha256_file(source_manifest),
        "train_annotations": str(split_root / "train.json"),
        "routes": ["first", "second", "balanced_hash_joint"],
        "trainable_parameters": "Qwen language model and lm_head only",
        "frozen_seek_sidecar": ["VGGT", "geometry_merger", "cross_attention"],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--order", choices=("self_then_seek", "seek_then_self"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--total-batch-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar-checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        write_manifest(
            args.run_root,
            order=args.order,
            steps=args.steps,
            total_batch_size=args.total_batch_size,
            world_size=args.world_size,
            qwen_checkpoint=args.qwen_checkpoint,
            vggt_checkpoint=args.vggt_checkpoint,
            sidecar_checkpoint=args.sidecar_checkpoint,
            split_root=args.split_root,
            repo_root=args.repo_root,
        )
    )


if __name__ == "__main__":
    main()
