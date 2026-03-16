"""Attention — Scaled Dot-Product Attention and Multi-Head Attention.

Part 1: Utility functions for causal mask and scaled dot-product attention.
Part 2: MultiHeadAttention layer with linear projections and head splitting.
"""

from __future__ import annotations

from tinynum import NDArray
from tinytorch.tensor import Tensor
from tinytorch.layer import Layer, Linear
from tinytorch import activations


# ================================================================
# Part 1: Causal Mask & Scaled Dot-Product Attention
# ================================================================


def create_causal_mask(seq_len: int) -> NDArray:
    """Creates a causal mask of shape [seq_len, seq_len].

    Mask value is 1.0 where the position should be masked (upper triangle),
    and 0.0 where the position is visible (lower triangle + diagonal).
    """
    raise NotImplementedError("TODO: E13")


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: NDArray | None = None
) -> Tensor:
    """Scaled Dot-Product Attention.

    Computes attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V,
    with optional masking.

    Args:
        Q: query tensor, shape [..., T, d_k]
        K: key tensor, shape [..., T, d_k]
        V: value tensor, shape [..., T, d_k]
        mask: optional mask (NDArray), 1.0 at positions to mask, None for no mask

    Returns:
        attention output, same shape as V
    """
    raise NotImplementedError("TODO: E13")


# ================================================================
# Part 2: Multi-Head Attention
# ================================================================


class MultiHeadAttention(Layer):
    """Multi-Head Attention layer.

    Splits the input into multiple heads, applies scaled dot-product
    attention independently per head, then merges and projects the output.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Creates a Multi-Head Attention layer.

        Args:
            embed_dim: total embedding dimension (must be divisible by num_heads)
            num_heads: number of attention heads
        """
        raise NotImplementedError("TODO: E13")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: project -> split heads -> attention -> merge -> project.

        Args:
            x: input tensor of shape [B, T, D]

        Returns:
            output tensor of shape [B, T, D]
        """
        raise NotImplementedError("TODO: E13")

    def parameters(self) -> list[Tensor]:
        raise NotImplementedError("TODO: E13")

    def children(self) -> list[Layer]:
        raise NotImplementedError("TODO: E13")
