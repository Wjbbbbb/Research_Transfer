"""Collect one reproducible four-frame SELF/SEEK counterfactual pair for M0.

Run this module once per sample.  It deliberately loads the routes in separate
model lifetimes so SELF cannot accidentally reuse VGGT allocated by SEEK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from .cache import append_records
from .inputs import prepare_route_inputs, validate_m0_frames
from .loading import load_seek_model, load_self_model
from .profiling import profile_call
from .types import CounterfactualRecord, Route


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def normalize_answer(answer: str) -> str:
    """M0's deliberately simple exact-match normalizer, not a benchmark scorer."""

    return re.sub(r"\s+", " ", answer.strip().casefold())


def checkpoint_id(checkpoint: Path) -> str:
    config = checkpoint / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config}")
    return f"{checkpoint.name}:{_sha256(config)[:12]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--split", required=True, help="Dataset split label recorded in the manifest.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--frame", action="append", required=True, help="Ordered full-frame path; provide exactly four.")
    parser.add_argument("--frame-id", action="append", required=True, type=int, help="Ordered integer frame ID; provide exactly four.")
    parser.add_argument("--reference-answer", help="Optional gold answer for teacher-forced loss and exact-match correctness.")
    parser.add_argument("--output", required=True, help="New JSONL destination for the SELF/SEEK pair.")
    parser.add_argument("--manifest", required=True, help="New JSON manifest destination for run provenance.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def _load_model(route: Route, checkpoint: Path, device: str):
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True, "device_map": device}
    return load_self_model(checkpoint, **kwargs) if route is Route.SELF else load_seek_model(checkpoint, **kwargs)


def _run_route(args: argparse.Namespace, route: Route, checkpoint: Path) -> CounterfactualRecord:
    model = _load_model(route, checkpoint, args.device)
    model.eval()
    device = next(model.parameters()).device
    try:
        inputs = prepare_route_inputs(
            checkpoint,
            args.frame,
            args.question,
            device=device,
            include_geometry=route is Route.SEEK,
        )
        prompt_length = inputs["input_ids"].shape[1]
        processor = AutoProcessor.from_pretrained(checkpoint)

        def generate() -> torch.Tensor:
            with torch.inference_mode():
                return model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        generated, profile = profile_call(generate, cuda_device=device)
        answer = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()
        loss: float | None = None
        correct: bool | None = None
        if args.reference_answer is not None:
            labeled_inputs = prepare_route_inputs(
                checkpoint,
                args.frame,
                args.question,
                device=device,
                include_geometry=route is Route.SEEK,
                target_answer=args.reference_answer,
            )
            with torch.inference_mode():
                loss = float(model(**labeled_inputs).loss.item())
            correct = normalize_answer(answer) == normalize_answer(args.reference_answer)
        return CounterfactualRecord(
            sample_id=args.sample_id,
            route=route,
            frame_ids=tuple(args.frame_id),
            answer=answer,
            loss=loss,
            correct=correct,
            profile=profile,
            checkpoint_id=checkpoint_id(checkpoint),
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _manifest(args: argparse.Namespace, checkpoint: Path, records: list[CounterfactualRecord]) -> dict[str, Any]:
    frame_paths = [Path(path) for path in args.frame]
    return {
        "schema_version": "dualgeo.m0.manifest.v1",
        "sample_id": args.sample_id,
        "split": args.split,
        "question": args.question,
        "frame_ids": args.frame_id,
        "frame_files": [{"path": str(path), "sha256": _sha256(path)} for path in frame_paths],
        "checkpoint": {"path": str(checkpoint), "id": checkpoint_id(checkpoint)},
        "routes": [record.route.value for record in records],
        "profiles": {record.route.value: record.profile.__dict__ for record in records},
        "reference_answer_provided": args.reference_answer is not None,
        "command": sys.argv,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "torch_version": torch.__version__,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": args.device,
            "name": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    validate_m0_frames(args.frame, args.frame_id)
    checkpoint = Path(args.checkpoint)
    output, manifest = Path(args.output), Path(args.manifest)
    if output.exists() or manifest.exists():
        raise FileExistsError("M0 collector refuses to append or overwrite an existing output/manifest")
    records = [_run_route(args, Route.SELF, checkpoint), _run_route(args, Route.SEEK, checkpoint)]
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    append_records(output, records)
    manifest.write_text(json.dumps(_manifest(args, checkpoint, records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": str(output), "manifest": str(manifest), "routes": [record.route.value for record in records]}))


if __name__ == "__main__":
    main()
