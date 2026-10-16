"""Route profiling with correct CUDA synchronization semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar
import time

import torch

from .types import InferenceProfile

P = ParamSpec("P")
T = TypeVar("T")


def profile_call(
    fn: Callable[P, T],
    *args: P.args,
    cuda_device: torch.device | str | int | None = None,
    **kwargs: P.kwargs,
) -> tuple[T, InferenceProfile]:
    """Run ``fn`` once and return its result plus end-to-end execution metrics.

    CUDA kernels are asynchronous, so synchronize immediately before and after
    the call on ``cuda_device``.  Supplying the model device is mandatory for a
    non-default GPU; otherwise CUDA's current device is used. CPU-only runs
    deliberately report ``peak_memory_mb=None`` rather than a misleading
    process-level number.
    """

    cuda_available = torch.cuda.is_available()
    device = None
    if cuda_available:
        device = torch.device(cuda_device) if cuda_device is not None else torch.cuda.current_device()
    if cuda_available:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    result = fn(*args, **kwargs)

    if cuda_available:
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mb = None

    profile = InferenceProfile(
        wall_time_ms=(time.perf_counter() - start) * 1_000,
        peak_memory_mb=peak_memory_mb,
    )
    return result, profile
