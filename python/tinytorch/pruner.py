"""Pruner — weight pruning utilities for model compression."""

from __future__ import annotations

from tinytorch.tensor import Tensor
from tinytorch.layer import Layer


def magnitude_prune(weight: Tensor, sparsity: float) -> None:
    """Magnitude pruning: zero out the smallest weights by absolute value.

    Args:
        weight: the weight tensor to prune (modified in-place).
        sparsity: fraction of weights to prune, in [0.0, 1.0].
    """
    raise NotImplementedError("TODO: E17")


def measure_sparsity(model: Layer) -> float:
    """Measure the sparsity of a model (percentage of zero-valued parameters).

    Only counts parameters with ndim >= 2 (skips biases).

    Args:
        model: the model to measure.

    Returns:
        Sparsity percentage in [0.0, 100.0].
    """
    raise NotImplementedError("TODO: E17")
