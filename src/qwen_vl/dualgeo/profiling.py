"""Route profiling with correct CUDA synchronization semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar
import time

import torch

from .types import InferenceProfile

P = ParamSpec("P")
T = TypeVar("T")


def profile_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> tuple[T, InferenceProfile]:
    """Run ``fn`` once and return its result plus end-to-end execution metrics.

    CUDA kernels are asynchronous, so synchronize immediately before and after
    the call.  CPU-only runs deliberately report ``peak_memory_mb=None`` rather
    than a misleading process-level number.
    """

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = fn(*args, **kwargs)

    if cuda_available:
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_memory_mb = None

    profile = InferenceProfile(
        wall_time_ms=(time.perf_counter() - start) * 1_000,
        peak_memory_mb=peak_memory_mb,
    )
    return result, profile
