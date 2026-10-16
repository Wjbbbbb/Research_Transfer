"""Stable, serializable types shared by DualGeo collection and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Route(str, Enum):
    """The only routes supported during M0."""

    SELF = "self"
    SEEK = "seek"


@dataclass(frozen=True)
class InferenceProfile:
    """Wall-clock and CUDA allocation measurements for a single route call."""

    wall_time_ms: float
    peak_memory_mb: float | None

    def __post_init__(self) -> None:
        if self.wall_time_ms < 0:
            raise ValueError("wall_time_ms must be non-negative")
        if self.peak_memory_mb is not None and self.peak_memory_mb < 0:
            raise ValueError("peak_memory_mb must be non-negative or None")


@dataclass(frozen=True)
class CounterfactualRecord:
    """One route result for one fixed sample and frame set.

    A SELF and SEEK record with the same ``sample_id`` and ``frame_ids`` form
    the training-time counterfactual pair used by later gate training.  M0
    intentionally stores metrics and answers only: VGGT tokens, pose, or other
    post-call features are forbidden here so they cannot leak into the gate.
    """

    sample_id: str
    route: Route
    frame_ids: tuple[int, ...]
    answer: str
    loss: float | None
    correct: bool | None
    profile: InferenceProfile
    checkpoint_id: str
    schema_version: str = "dualgeo.m0.v1"

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.frame_ids:
            raise ValueError("frame_ids must not be empty")
        if tuple(sorted(set(self.frame_ids))) != self.frame_ids:
            raise ValueError("frame_ids must be unique and sorted")
        if self.loss is not None and self.loss < 0:
            raise ValueError("loss must be non-negative or None")
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["route"] = self.route.value
        payload["frame_ids"] = list(self.frame_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CounterfactualRecord":
        expected = {
            "sample_id",
            "route",
            "frame_ids",
            "answer",
            "loss",
            "correct",
            "profile",
            "checkpoint_id",
            "schema_version",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown or missing:
            raise ValueError(
                f"Counterfactual record schema mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if payload["schema_version"] != "dualgeo.m0.v1":
            raise ValueError(f"Unsupported schema version: {payload['schema_version']}")
        return cls(
            sample_id=payload["sample_id"],
            route=Route(payload["route"]),
            frame_ids=tuple(payload["frame_ids"]),
            answer=payload["answer"],
            loss=payload["loss"],
            correct=payload["correct"],
            profile=InferenceProfile(**payload["profile"]),
            checkpoint_id=payload["checkpoint_id"],
            schema_version=payload["schema_version"],
        )
