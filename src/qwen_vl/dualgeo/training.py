"""Small, auditable M1 helpers shared by the standard training entry point.

The helpers deliberately do not implement a gate.  They only make the route
presented to the existing SFT loader explicit and keep the SEEK-only modules
outside of the A/B trainable-parameter set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .types import Route

_SIDECAR_PREFIXES = ("model.geometry_merger.",)


def route_for_sample(sample_id: str, schedule: str, seed: int) -> Route:
    """Return a deterministic M1 route for one sample.

    ``balanced_hash`` is used only for the mandatory joint-consolidation
    phase.  It does not inspect question contents or model outputs, and is
    therefore stable across workers and resumes.
    """

    if schedule == "self":
        return Route.SELF
    if schedule == "seek":
        return Route.SEEK
    if schedule == "balanced_hash":
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        return Route.SELF if digest[0] % 2 == 0 else Route.SEEK
    raise ValueError(f"Unsupported DualGeo route schedule: {schedule!r}")


def load_frozen_seek_sidecar(model: Any, checkpoint: str | Path) -> dict[str, int]:
    """Load only static GeoThinker fusion weights into an already-built model.

    The raw Qwen checkpoint remains the common initialisation for A/B.  VGGT
    itself is loaded separately from its declared pretrained path; this helper
    transfers only the trained geometry merger and per-layer cross-attention
    modules from the fixed GeoThinker reference checkpoint.
    """

    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("DualGeo SEEK sidecar loading requires safetensors") from exc

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    selected = [
        name
        for name in weight_map
        if name.startswith(_SIDECAR_PREFIXES) or ".cross_attn." in name
    ]
    if not selected:
        raise ValueError(f"No SEEK sidecar tensors found in {checkpoint}")

    tensors: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard, names in by_shard.items():
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as handle:
            tensors.update({name: handle.get_tensor(name) for name in names})

    expected = set(model.state_dict())
    unexpected = sorted(set(tensors) - expected)
    if unexpected:
        raise ValueError(
            "SEEK sidecar architecture mismatch; unexpected tensors include "
            f"{unexpected[:3]}"
        )
    result = model.load_state_dict(tensors, strict=False)
    missing_sidecar = [
        name
        for name in expected
        if (name.startswith(_SIDECAR_PREFIXES) or ".cross_attn." in name) and name in result.missing_keys
    ]
    if missing_sidecar:
        raise ValueError(
            "SEEK sidecar checkpoint did not initialise all required tensors; "
            f"missing {missing_sidecar[:3]}"
        )
    return {"loaded": len(tensors), "geometry_merger": sum(name.startswith(_SIDECAR_PREFIXES) for name in tensors), "cross_attn": sum(".cross_attn." in name for name in tensors)}


def freeze_seek_sidecar(model: Any) -> int:
    """Freeze non-Qwen SEEK modules, returning the number of frozen tensors."""

    frozen = 0
    for name, parameter in model.named_parameters():
        if (
            "geometry_encoder" in name
            or name.startswith("model.geometry_merger.")
            or name.startswith("model.feature_fusion.")
            or ".cross_attn." in name
        ):
            parameter.requires_grad = False
            frozen += 1
    return frozen
