"""test_e09.py — E09 Training Loop test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor
from tinytorch.trainer import Trainer
from tinytorch.layer import Layer
from tinytorch import losses
from tinytorch.optimizer import SGD


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def get_at(t: Tensor, i: int) -> float:
    """Get element at index i of a 1D tensor."""
    return t.data.get(i)


def get_at_2d(t: Tensor, row: int, col: int) -> float:
    """Get element at (row, col) of a 2D tensor."""
    return t.data.get(row, col)


class SimpleModel(Layer):
    """Test model: pred = input + w."""

    def __init__(self):
        super().__init__()
        self.w = Tensor.from_array([2.0], 1)
        self.w.requires_grad = True

    def forward(self, x):
        return x.add(self.w)

    def parameters(self):
        return [self.w]


def main() -> None:
    # =============================================================
    # Test 1–4: Cosine schedule
    # =============================================================
    lr0 = Trainer.cosine_schedule(0, 100, 0.1, 0.001)
    emit("cosine_at_start", float_str(lr0))

    lr50 = Trainer.cosine_schedule(50, 100, 0.1, 0.001)
    emit("cosine_at_middle", float_str(lr50))

    lr100 = Trainer.cosine_schedule(100, 100, 0.1, 0.001)
    emit("cosine_at_end", float_str(lr100))

    lr25 = Trainer.cosine_schedule(25, 100, 0.1, 0.001)
    emit("cosine_at_quarter", float_str(lr25))

    # =============================================================
    # Test 5–7: clipGradNorm — clipping applied
    # grad = [3.0, 4.0] → norm = 5.0, maxNorm = 2.5
    # After clip: grad = [1.5, 2.0], returns 5.0
    # =============================================================
    p = Tensor.from_array([0.0, 0.0], 2)
    p.requires_grad = True
    p.grad = Tensor.from_array([3.0, 4.0], 2)

    norm = Trainer.clip_grad_norm([p], 2.5)
    emit("clip_returns_norm", float_str(norm))
    emit("clip_after_grad0", float_str(get_at(p.grad, 0)))
    emit("clip_after_grad1", float_str(get_at(p.grad, 1)))

    # =============================================================
    # Test 8: clipGradNorm — no clipping needed
    # grad = [1.0, 0.0] → norm = 1.0, maxNorm = 5.0
    # =============================================================
    p = Tensor.from_array([0.0, 0.0], 2)
    p.requires_grad = True
    p.grad = Tensor.from_array([1.0, 0.0], 2)

    Trainer.clip_grad_norm([p], 5.0)
    emit("clip_no_clip_grad0", float_str(get_at(p.grad, 0)))

    # =============================================================
    # Test 9: Accuracy — all correct
    # pred argmax = [0, 1], target argmax = [0, 1] → 1.0
    # =============================================================
    pred = Tensor.from_array([0.9, 0.1, 0.2, 0.8], 2, 2)
    target = Tensor.from_array([1.0, 0.0, 0.0, 1.0], 2, 2)
    acc = Trainer.accuracy(pred, target)
    emit("accuracy_perfect", float_str(acc))

    # =============================================================
    # Test 10: Accuracy — half correct
    # pred argmax = [0, 0], target argmax = [0, 1] → 0.5
    # =============================================================
    pred = Tensor.from_array([0.9, 0.1, 0.8, 0.2], 2, 2)
    target = Tensor.from_array([1.0, 0.0, 0.0, 1.0], 2, 2)
    acc = Trainer.accuracy(pred, target)
    emit("accuracy_partial", float_str(acc))

    # =============================================================
    # Test 11–12: Training step
    # Simple model: pred = input + w, MSE loss, SGD lr=0.1
    # w starts at 2.0, input = [1.0], target = [5.0]
    # loss1 = (1+2-5)^2 = 4.0 > 0
    # After 10 steps, loss decreases
    # =============================================================
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, losses.mse)

    x = Tensor.from_array([1.0], 1)
    y = Tensor.from_array([5.0], 1)

    loss1 = trainer.train_step(x, y)
    emit("train_step_positive", str(loss1 > 0).lower())

    # Run 9 more steps
    for _ in range(9):
        trainer.train_step(x, y)
    loss10 = trainer.train_step(x, y)
    emit("train_step_decreasing", str(loss10 < loss1).lower())


if __name__ == "__main__":
    main()
