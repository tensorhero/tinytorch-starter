"""Profiler — performance analysis for tinytorch models."""

from __future__ import annotations

from tinytorch.layer import Layer, Linear


def count_params(model: Layer) -> int:
    """Count the total number of trainable parameters in a model.

    Recursively traverses all children layers.
    """
    raise NotImplementedError("TODO: E17")


def count_flops(model: Layer, input_shape: tuple[int, ...]) -> int:
    """Estimate the total FLOPs for a forward pass.

    Only counts Linear layer FLOPs: 2 × in_features × out_features.
    Recursively traverses all children layers.

    Args:
        model: the model to profile.
        input_shape: the input tensor shape, e.g. (batch_size, seq_len, dim).

    Returns:
        Estimated FLOPs.
    """
    raise NotImplementedError("TODO: E17")
