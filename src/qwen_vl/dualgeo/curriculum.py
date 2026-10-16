"""Immutable M1 experiment order contracts for SELF/SEEK curriculum studies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import Route


class CurriculumOrder(str, Enum):
    SELF_THEN_SEEK = "self_then_seek"
    SEEK_THEN_SELF = "seek_then_self"


@dataclass(frozen=True)
class CurriculumPlan:
    """The two phase orders plus a shared consolidation phase.

    Consolidation is mandatory: without it, the second phase mainly measures
    catastrophic forgetting rather than a usable final dual-route system.
    """

    order: CurriculumOrder
    phase_one: Route
    phase_two: Route
    consolidation_routes: tuple[Route, Route] = (Route.SELF, Route.SEEK)

    def __post_init__(self) -> None:
        if self.phase_one is self.phase_two:
            raise ValueError("the two curriculum phases must use different routes")
        if set(self.consolidation_routes) != {Route.SELF, Route.SEEK}:
            raise ValueError("consolidation must contain exactly SELF and SEEK")


def build_curriculum(order: CurriculumOrder | str) -> CurriculumPlan:
    order = CurriculumOrder(order)
    if order is CurriculumOrder.SELF_THEN_SEEK:
        return CurriculumPlan(order, Route.SELF, Route.SEEK)
    return CurriculumPlan(order, Route.SEEK, Route.SELF)
