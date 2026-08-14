"""Numerically safe primitives for contrastive-feature training."""

from __future__ import annotations

import torch


def safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean selected values, or a differentiable zero for an empty selection."""
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def sample_consistent_pairs(
    consistent: torch.Tensor,
    random_values: torch.Tensor,
    target_count: torch.Tensor | float | int,
) -> torch.Tensor:
    """Randomly subsample a boolean population without division by zero."""
    count = int(torch.count_nonzero(consistent).item())
    if count == 0:
        return torch.zeros_like(consistent, dtype=torch.bool)
    probability = min(1.0, float(target_count) / count)
    return torch.logical_and(consistent, random_values < probability)
