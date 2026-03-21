"""Optimizer — base class and concrete optimizers for parameter updates.

Provides:
- Optimizer: abstract base with zero_grad() and step()
- SGD: stochastic gradient descent with optional momentum
- Adam: adaptive moment estimation
- AdamW: Adam with decoupled weight decay
"""

from __future__ import annotations

from tinynum import NDArray
from tinytorch.tensor import Tensor


class Optimizer:
    """Abstract base class for all optimizers."""

    # ================================================================
    # E08 — Optimizers
    # ================================================================

    def __init__(self, params: list[Tensor]) -> None:
        """Creates an optimizer for the given parameters."""
        pass  # TODO: E08

    def zero_grad(self) -> None:
        """Clears the gradients of all parameters (sets grad to None)."""
        raise NotImplementedError("TODO: E08")

    def step(self) -> None:
        """Performs a single optimization step (parameter update)."""
        raise NotImplementedError("TODO: E08")


class SGD(Optimizer):
    """SGD with optional momentum.

    Without momentum: param -= lr * grad
    With momentum: v = momentum * v + grad; param -= lr * v
    """

    def __init__(self, params: list[Tensor], lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(params)
        # TODO: E08

    def step(self) -> None:
        raise NotImplementedError("TODO: E08")


class Adam(Optimizer):
    """Adam — Adaptive Moment Estimation.

    Maintains first and second moment estimates with bias correction.
    Default: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params)
        # TODO: E08

    def step(self) -> None:
        raise NotImplementedError("TODO: E08")


class AdamW(Optimizer):
    """AdamW — Adam with decoupled weight decay.

    Applies weight decay independently before gradient update.
    Default: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params)
        # TODO: E08

    def step(self) -> None:
        raise NotImplementedError("TODO: E08")
