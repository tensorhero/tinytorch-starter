"""Activations — element-wise nonlinear activation functions.

All functions accept a Tensor and return a new Tensor.
No learnable parameters. Forward-only in E02;
backward implementations will be added in E06.
"""

from __future__ import annotations

from tinytorch.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """ReLU activation: f(x) = max(0, x)."""
    raise NotImplementedError("TODO: E02")


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))."""
    raise NotImplementedError("TODO: E02")


def tanh(x: Tensor) -> Tensor:
    """Tanh activation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))."""
    raise NotImplementedError("TODO: E02")


def gelu(x: Tensor) -> Tensor:
    """GELU activation (tanh approximation):
    f(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    raise NotImplementedError("TODO: E02")


def softmax(x: Tensor, axis: int) -> Tensor:
    """Softmax activation: f(x_i) = exp(x_i - max) / sum(exp(x_j - max)).

    Numerically stable: subtracts the max before computing exp.
    """
    raise NotImplementedError("TODO: E02")


# ================================================================
# E06 — More Backward Ops
# ================================================================
# When x.requires_grad is True, relu(), sigmoid(), tanh(), and gelu()
# should create the corresponding Backward function (ReLUBackward,
# SigmoidBackward, TanhBackward, GELUBackward), call fn.forward(),
# and set result.requires_grad = True, result.grad_fn = fn.
# softmax does NOT need backward here (cross_entropy uses fusion).
