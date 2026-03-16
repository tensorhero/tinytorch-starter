"""GPT — complete GPT language model with autoregressive generation.

Stacks Embedding, PositionalEncoding, TransformerBlock layers,
a final LayerNorm, and a linear head to produce token logits.
Includes a generate method for autoregressive text generation.
"""

from __future__ import annotations

from tinynum import NDArray
from tinytorch.tensor import Tensor
from tinytorch.layer import Layer, Linear
from tinytorch.embedding import Embedding, PositionalEncoding
from tinytorch.transformer_block import LayerNorm, Block
from tinytorch.tokenizer import CharTokenizer
from tinytorch import activations


class GPT(Layer):
    """Complete GPT language model."""

    # ================================================================
    # E15 — GPT & Generate
    # ================================================================

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: E15")

    def forward(self, token_ids: list[int]) -> Tensor:
        """Forward pass: token IDs -> logits [seqLen, vocabSize]."""
        raise NotImplementedError("TODO: E15")

    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Autoregressive text generation."""
        raise NotImplementedError("TODO: E15")

    def parameters(self) -> list[Tensor]:
        raise NotImplementedError("TODO: E15")

    def children(self) -> list[Layer]:
        raise NotImplementedError("TODO: E15")
