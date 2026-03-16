"""KVCache — caches Key and Value tensors for efficient autoregressive generation.

Stores and concatenates Key/Value NDArrays along a specified axis,
avoiding redundant computation during Transformer inference.
"""

from __future__ import annotations

from tinynum import NDArray


class KVCache:
    """Caches key/value tensors, concatenating along a specified axis."""

    # ================================================================
    # E16 — Quantization & KV Cache
    # ================================================================

    def __init__(self, axis: int) -> None:
        raise NotImplementedError("TODO: E16")

    def update(self, new_keys: NDArray, new_values: NDArray) -> None:
        """Append new keys and values to the cache."""
        raise NotImplementedError("TODO: E16")

    def get_keys(self) -> NDArray:
        """Return the accumulated keys."""
        raise NotImplementedError("TODO: E16")

    def get_values(self) -> NDArray:
        """Return the accumulated values."""
        raise NotImplementedError("TODO: E16")

    def current_len(self) -> int:
        """Return the current cached sequence length."""
        raise NotImplementedError("TODO: E16")

    def reset(self) -> None:
        """Clear all stored keys and values."""
        raise NotImplementedError("TODO: E16")
