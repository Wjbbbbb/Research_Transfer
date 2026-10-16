"""Small dependency-free calibration metrics for M1 evaluation."""

from __future__ import annotations

import torch


def brier_score(probability: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Mean squared error of predicted error probability and binary error."""

    _validate_probability_and_error(probability, error)
    return ((probability - error.to(probability.dtype)) ** 2).mean()


def expected_calibration_error(probability: torch.Tensor, error: torch.Tensor, bins: int = 15) -> torch.Tensor:
    """Equal-width expected calibration error for error-risk probabilities."""

    _validate_probability_and_error(probability, error)
    if bins <= 0:
        raise ValueError("bins must be positive")
    probability, error = probability.flatten(), error.flatten().to(probability.dtype)
    total = probability.numel()
    ece = probability.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probability >= lower) & ((probability < upper) if index + 1 < bins else (probability <= upper))
        if mask.any():
            ece = ece + mask.float().mean() * (probability[mask].mean() - error[mask].mean()).abs()
    return ece


def risk_coverage_curve(probability: torch.Tensor, error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return coverage and empirical error rate after accepting lowest-risk rows."""

    _validate_probability_and_error(probability, error)
    ordered_error = error.flatten()[torch.argsort(probability.flatten())].to(probability.dtype)
    coverage = torch.arange(1, ordered_error.numel() + 1, device=probability.device, dtype=probability.dtype)
    coverage = coverage / ordered_error.numel()
    return coverage, torch.cumsum(ordered_error, dim=0) / torch.arange(
        1, ordered_error.numel() + 1, device=probability.device, dtype=probability.dtype
    )


def _validate_probability_and_error(probability: torch.Tensor, error: torch.Tensor) -> None:
    if probability.numel() == 0 or probability.shape != error.shape:
        raise ValueError("probability and error must have the same non-empty shape")
    if torch.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must lie in [0, 1]")
    if error.dtype != torch.bool and torch.any((error != 0) & (error != 1)):
        raise ValueError("error must be boolean or contain only 0/1")
