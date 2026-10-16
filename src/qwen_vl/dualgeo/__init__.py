"""DualGeo primitives for separating and training SELF/SEEK routes."""

from .config import FastAdapterConfig
from .curriculum import CurriculumOrder, build_curriculum
from .fast import FastRouteState, FastSpatialAdapter
from .types import CounterfactualRecord, Route

__all__ = [
    "CounterfactualRecord",
    "CurriculumOrder",
    "FastAdapterConfig",
    "FastRouteState",
    "FastSpatialAdapter",
    "Route",
    "build_curriculum",
]
