"""Losses — loss functions for training neural networks.

All functions return a scalar Tensor of shape [1].
Forward-only in E04; backward implementations will be added in E06.
"""

from __future__ import annotations

from tinynum import NDArray
from tinytorch.tensor import Tensor


def log_softmax(x: Tensor, axis: int) -> Tensor:
    """Numerically stable log-softmax along the given axis.

    logSoftmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))
    """
    raise NotImplementedError("TODO: E04")


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error: mean((pred - target)^2).

    Returns:
        Scalar Tensor of shape [1].
    """
    raise NotImplementedError("TODO: E04")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss with one-hot encoded targets.

    CE = -mean(sum(targets * logSoftmax(logits), axis=classes))

    Args:
        logits: raw scores, shape [batch, num_classes].
        targets: one-hot encoded, shape [batch, num_classes].

    Returns:
        Scalar Tensor of shape [1].
    """
    raise NotImplementedError("TODO: E04")


# ================================================================
# E06 — More Backward Ops
# ================================================================
# cross_entropy: when logits.requires_grad, create CrossEntropyBackward,
#   call fn.forward(), set result.requires_grad = True, result.grad_fn = fn.
#   Uses fused gradient: dL/dlogits = (softmax(logits) - targets) / batch_size.
# mse: rewrite to use Tensor ops (sub → mul → reshape → mean) so the
#   computation graph is recorded automatically. No MSEBackward needed.
