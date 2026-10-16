"""Identical multi-image prompt preparation for DualGeo M0 routes.

The Qwen processor prepares language/vision inputs, while GeoThinker's VGGT
expects its own per-frame tensor.  This module creates both from exactly the
same ordered image paths.  It deliberately contains no routing policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from qwen_vl.data.utils import load_and_preprocess_images


def validate_m0_frames(image_paths: Sequence[str | Path], frame_ids: Sequence[int] | None = None) -> None:
    """Require M0's fixed, ordered four-full-frame protocol."""

    if len(image_paths) != 4:
        raise ValueError(f"M0 requires exactly 4 full frames, got {len(image_paths)}")
    if frame_ids is not None:
        if len(frame_ids) != 4 or tuple(sorted(set(frame_ids))) != tuple(frame_ids):
            raise ValueError("M0 frame_ids must contain exactly 4 unique, ascending integers")


def _load_images(image_paths: Sequence[str | Path]) -> list[Image.Image]:
    paths = [Path(path) for path in image_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M0 frame files not found: {missing}")
    return [Image.open(path).convert("RGB") for path in paths]


def prepare_route_inputs(
    checkpoint_path: str | Path,
    image_paths: Sequence[str | Path],
    question: str,
    *,
    device: torch.device | str,
    include_geometry: bool,
    target_answer: str | None = None,
) -> dict[str, Any]:
    """Prepare a single-sample four-frame question for SELF or SEEK.

    ``include_geometry=False`` never constructs the VGGT-specific tensor.
    ``include_geometry=True`` adds one ``[4, C, H, W]`` tensor in the format
    expected by GeoThinker's ``geometry_encoder_inputs`` API.
    """

    validate_m0_frames(image_paths)
    if not question.strip():
        raise ValueError("question must not be empty")

    images = _load_images(image_paths)
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": question})
    user_message = {"role": "user", "content": content}
    prompt = processor.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    result: dict[str, Any] = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }

    if include_geometry:
        # This is GeoThinker's established VGGT preprocessing, intentionally
        # separate from the processor's Qwen-vision representation.
        geometry_frames = load_and_preprocess_images(images)
        if geometry_frames.shape[0] != len(images):
            raise RuntimeError("VGGT frame preprocessing changed the M0 frame count")
        result["geometry_encoder_inputs"] = [geometry_frames.to(device)]

    if target_answer is not None:
        if not target_answer.strip():
            raise ValueError("target_answer must not be empty when computing loss")
        full_prompt = processor.apply_chat_template(
            [user_message, {"role": "assistant", "content": target_answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_batch = processor(text=[full_prompt], images=images, return_tensors="pt", padding=True)
        prompt_ids = batch["input_ids"]
        full_ids = full_batch["input_ids"]
        prompt_length = prompt_ids.shape[1]
        if full_ids.shape[1] <= prompt_length or not torch.equal(full_ids[:, :prompt_length], prompt_ids):
            raise RuntimeError("assistant target does not extend the M0 generation prompt")
        labels = full_ids.clone()
        labels[:, :prompt_length] = -100
        result = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in full_batch.items()
        }
        result["labels"] = labels.to(device)
        if include_geometry:
            result["geometry_encoder_inputs"] = [geometry_frames.to(device)]
    return result
