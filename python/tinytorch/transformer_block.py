"""TransformerBlock — LayerNorm, MLP, and Pre-norm Transformer Block.

Part 1: LayerNorm — layer normalization with learnable gamma/beta.
Part 2: MLP — two-layer feed-forward network with GELU activation.
Part 3: Block — Pre-norm architecture with residual connections.
"""

from __future__ import annotations

from tinynum import NDArray
from tinytorch.tensor import Tensor
from tinytorch.layer import Layer, Linear
from tinytorch import activations
from tinytorch.attention import MultiHeadAttention


# ================================================================
# Part 1: LayerNorm
# ================================================================


class LayerNorm(Layer):
    """Layer Normalization.

    Normalizes the input along the last dimension:
    output = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        raise NotImplementedError("TODO: E14")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: E14")

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]


# ================================================================
# Part 2: MLP
# ================================================================


class MLP(Layer):
    """MLP (Multi-Layer Perceptron) with 4x expansion.

    Structure: fc1 (D -> 4D) -> GELU -> fc2 (4D -> D)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: E14")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: E14")

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def children(self) -> list[Layer]:
        return [self.fc1, self.fc2]


# ================================================================
# Part 3: Block (TransformerBlock)
# ================================================================


class Block(Layer):
    """Pre-norm Transformer Block.

    Architecture:
    x = x + attn(ln1(x))
    x = x + mlp(ln2(x))
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: E14")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("TODO: E14")

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params

    def children(self) -> list[Layer]:
        return [self.ln1, self.attn, self.ln2, self.mlp]
