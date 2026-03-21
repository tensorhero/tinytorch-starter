"""Trainer — orchestrates the standard training loop.

Encapsulates forward → loss → backward → step → zero_grad,
plus utilities for learning rate scheduling, gradient clipping,
and accuracy evaluation.
"""

from __future__ import annotations

import math
from typing import Callable

from tinynum import NDArray
from tinytorch.tensor import Tensor


class Trainer:
    """Training loop orchestrator."""

    # ================================================================
    # E09 — Training Loop
    # ================================================================

    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        raise NotImplementedError("TODO: E09")

    def train_step(self, x: Tensor, y: Tensor) -> float:
        """Single batch training step: forward → loss → backward → step → zero_grad."""
        raise NotImplementedError("TODO: E09")

    @staticmethod
    def accuracy(pred: Tensor, target: Tensor) -> float:
        """Computes classification accuracy (argmax comparison)."""
        raise NotImplementedError("TODO: E09")

    @staticmethod
    def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
        """Clips gradient norm, returns original total norm."""
        raise NotImplementedError("TODO: E09")

    @staticmethod
    def cosine_schedule(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
        """Cosine annealing learning rate schedule."""
        raise NotImplementedError("TODO: E09")
