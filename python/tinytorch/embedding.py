"""Embedding and PositionalEncoding for TinyTorch.

Provides:
- Embedding: token ID → dense vector lookup with autograd support
- EmbeddingBackward: gradient computation via scatter-add
- PositionalEncoding: sinusoidal (fixed) or learned position encoding
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinynum import NDArray
from tinytorch.tensor import Tensor
from tinytorch.function import Function

if TYPE_CHECKING:
    pass


class EmbeddingBackward(Function):
    """Backward function for embedding lookup.

    Accumulates gradients back to weight matrix using scatter-add.
    """

    def __init__(self, indices: list[int], vocab_size: int, embed_dim: int) -> None:
        raise NotImplementedError("TODO: E12")

    def forward(self, *inputs: Tensor) -> list[Tensor]:
        raise NotImplementedError("TODO: E12")

    def backward(self, grad_output: NDArray) -> list[NDArray]:
        raise NotImplementedError("TODO: E12")

    def get_inputs(self) -> list[Tensor]:
        raise NotImplementedError("TODO: E12")


class Embedding:
    """Embedding — maps token IDs to dense vectors via table lookup.

    Weight shape: [vocab_size, embed_dim]. Xavier initialization.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        raise NotImplementedError("TODO: E12")

    def forward(self, indices: list[int]) -> Tensor:
        """Looks up embedding vectors for the given token indices."""
        raise NotImplementedError("TODO: E12")

    def parameters(self) -> list[Tensor]:
        """Returns trainable parameters (the weight matrix)."""
        raise NotImplementedError("TODO: E12")


class PositionalEncoding:
    """PositionalEncoding — adds position information to token embeddings.

    Modes:
    - "sinusoidal": fixed sin/cos table (no trainable parameters)
    - "learned": trainable position embedding (uses Embedding internally)
    """

    def __init__(self, mode: str, max_len: int, dim: int) -> None:
        raise NotImplementedError("TODO: E12")

    @staticmethod
    def build_sinusoidal_table(max_len: int, dim: int) -> NDArray:
        """Builds sinusoidal positional encoding table.

        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
        """
        raise NotImplementedError("TODO: E12")

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """Adds positional encoding to token embeddings."""
        raise NotImplementedError("TODO: E12")

    def parameters(self) -> list[Tensor]:
        """Returns trainable parameters (empty for sinusoidal, one for learned)."""
        raise NotImplementedError("TODO: E12")
