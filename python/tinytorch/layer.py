"""Layer — abstract base class and concrete neural network layers.

Provides:
- Layer: abstract base with train/eval mode switching
- Linear: fully connected layer (y = x @ W.T + b)
- Dropout: regularization via inverted dropout
- Sequential: chains multiple layers into a model
"""

from __future__ import annotations

import random
from typing import Optional

from tinynum import NDArray
from tinytorch.tensor import Tensor


class Layer:
    """Abstract base class for all neural network layers."""

    def __init__(self) -> None:
        self.training: bool = True

    # ================================================================
    # E03 — Linear Layer
    # ================================================================

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass (subclasses must implement)."""
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """Returns trainable parameters (default: empty)."""
        return []

    def children(self) -> list["Layer"]:
        """Returns child layers (default: empty)."""
        return []

    def train(self) -> None:
        """Recursively sets training mode."""
        raise NotImplementedError("TODO: E03")

    def eval(self) -> None:
        """Recursively sets evaluation mode."""
        raise NotImplementedError("TODO: E03")


class Linear(Layer):
    """Linear (fully connected) layer: y = x @ W.T + b.

    Weight shape: [out_features, in_features] (PyTorch convention).
    Uses LeCun initialization: W ~ N(0, 1/in_features), b = zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Creates a Linear layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            bias: whether to include a bias term
        """
        raise NotImplementedError("TODO: E03")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = x @ W.T + b."""
        raise NotImplementedError("TODO: E03")

    def parameters(self) -> list[Tensor]:
        """Returns [weight] or [weight, bias]."""
        raise NotImplementedError("TODO: E03")


class Dropout(Layer):
    """Dropout layer for regularization.

    Training: randomly zeros elements with probability p, scales by 1/(1-p).
    Evaluation: passes input through unchanged.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Creates a Dropout layer.

        Args:
            p: probability of zeroing each element (0.0 to 1.0)
        """
        raise NotImplementedError("TODO: E03")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: inverted dropout in training, identity in eval."""
        raise NotImplementedError("TODO: E03")

    # ================================================================
    # E06 — More Backward Ops
    # ================================================================
    # When x.requires_grad and self.training, create DropoutBackward,
    # call fn.forward(), set result.requires_grad = True, result.grad_fn = fn.
    # DropoutBackward reuses last_mask saved during forward.


class Sequential(Layer):
    """Sequential container — chains layers into a single model."""

    def __init__(self, *layers: Layer) -> None:
        """Creates a Sequential with the given layers in order."""
        raise NotImplementedError("TODO: E03")

    def forward(self, x: Tensor) -> Tensor:
        """Passes input through each layer sequentially."""
        raise NotImplementedError("TODO: E03")

    def parameters(self) -> list[Tensor]:
        """Collects parameters from all child layers."""
        raise NotImplementedError("TODO: E03")

    def children(self) -> list[Layer]:
        """Returns all child layers."""
        raise NotImplementedError("TODO: E03")
