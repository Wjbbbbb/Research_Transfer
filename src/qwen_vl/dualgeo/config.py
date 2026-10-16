"""Configuration objects for the VGGT-free M1 fast system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FastAdapterConfig:
    """Shape contract for :class:`FastSpatialAdapter`.

    The adapter consumes frozen Qwen token features.  It has no geometry model
    path or external-feature setting by construction.
    """

    hidden_size: int
    bottleneck_size: int = 512
    dropout: float = 0.0
    num_predicates: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_predicates < 0:
            raise ValueError("num_predicates must be non-negative")
