"""Explicit model loaders for DualGeo's physically separate M0 routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig


def build_self_config(checkpoint_path: str | Path):
    """Create a checkpoint-compatible config that cannot instantiate VGGT."""

    config = AutoConfig.from_pretrained(checkpoint_path)
    config.use_geometry_encoder = False
    if hasattr(config, "text_config"):
        config.text_config.geo_cross_attn = False
    return config


def load_self_model(checkpoint_path: str | Path, **kwargs: Any):
    """Load the Qwen backbone without a geometry encoder or fusion modules."""

    from qwen_vl.model.qwenvl3.modeling_qwen3_vl import (
        Qwen3VLForConditionalGenerationWithVGGT,
    )

    model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        checkpoint_path,
        config=build_self_config(checkpoint_path),
        **kwargs,
    )
    assert_self_model_has_no_vggt(model)
    return model


def load_seek_model(
    checkpoint_path: str | Path,
    geometry_encoder_path: str | None = None,
    **kwargs: Any,
):
    """Load the GeoThinker checkpoint and its frozen VGGT dependency for SEEK.

    GeoThinker checkpoints may already include ``geometry_encoder`` weights.  In
    that case leave ``geometry_encoder_path`` unset to avoid a redundant model
    download/reload; provide it only to deliberately override the embedded VGGT.
    """

    from qwen_vl.model.qwenvl3.modeling_qwen3_vl import (
        Qwen3VLForConditionalGenerationWithVGGT,
    )

    config = AutoConfig.from_pretrained(checkpoint_path)
    if not getattr(config, "use_geometry_encoder", False):
        raise ValueError("SEEK requires a checkpoint configured with a geometry encoder")
    load_kwargs = dict(kwargs)
    if geometry_encoder_path is not None:
        load_kwargs["geometry_encoder_path"] = geometry_encoder_path
    return Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        checkpoint_path,
        config=config,
        **load_kwargs,
    )


def assert_self_model_has_no_vggt(model: Any) -> None:
    """Fail closed if a supposedly fast model contains a geometry encoder."""

    forbidden = [name for name, _ in model.named_modules() if "geometry_encoder" in name]
    if forbidden:
        raise AssertionError(f"SELF model unexpectedly contains VGGT modules: {forbidden}")
