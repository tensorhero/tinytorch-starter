"""test_e04.py — E04 Loss Functions test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify your Losses implementations.
"""

import math
import os
import sys

# Ensure packages are importable regardless of how this script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor
from tinytorch.losses import log_softmax, mse, cross_entropy


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:

    # ===== MSE =====

    # Perfect prediction → loss = 0
    p1 = Tensor.from_array([1, 2, 3], 3)
    t1 = Tensor.from_array([1, 2, 3], 3)
    mse1 = mse(p1, t1)
    emit("mse_zero_loss", float_str(mse1.data.get(0)))

    # Basic: pred=[1,0], target=[0,1] → MSE=1.0
    p2 = Tensor.from_array([1, 0], 2)
    t2 = Tensor.from_array([0, 1], 2)
    mse2 = mse(p2, t2)
    emit("mse_basic", float_str(mse2.data.get(0)))

    # Known: diff=[-0.5,0.5,0.0,1.0] → MSE=0.375
    p3 = Tensor.from_array([2.5, 0.0, 2.0, 8.0], 4)
    t3 = Tensor.from_array([3.0, -0.5, 2.0, 7.0], 4)
    mse3 = mse(p3, t3)
    emit("mse_known", float_str(mse3.data.get(0)))

    # 2D: same data reshaped to [2,2] → same result
    p4 = Tensor.from_array([2.5, 0.0, 2.0, 8.0], 2, 2)
    t4 = Tensor.from_array([3.0, -0.5, 2.0, 7.0], 2, 2)
    mse4 = mse(p4, t4)
    emit("mse_2d", float_str(mse4.data.get(0)))

    # Shape: should be [1]
    emit("mse_shape", shape_str(mse3.shape()))

    # ===== logSoftmax =====

    # logSoftmax([1,2,3], axis=0) → first element ≈ -2.4076
    lx1 = Tensor.from_array([1, 2, 3], 3)
    ls1 = log_softmax(lx1, 0)
    emit("logsoftmax_val0", float_str(ls1.data.get(0)))

    # sum(exp(logSoftmax)) ≈ 1.0
    sum_exp = sum(math.exp(ls1.data.get(i)) for i in range(3))
    emit("logsoftmax_sum_exp", float_str(sum_exp))

    # Uniform: logSoftmax([1,1,1], axis=0)[0] = log(1/3) ≈ -1.0986
    lx2 = Tensor.from_array([1, 1, 1], 3)
    ls2 = log_softmax(lx2, 0)
    emit("logsoftmax_uniform", float_str(ls2.data.get(0)))

    # Numerical stability: large values don't overflow
    lx3 = Tensor.from_array([1000, 1001, 1002], 3)
    ls3 = log_softmax(lx3, 0)
    sum_exp_stable = sum(math.exp(ls3.data.get(i)) for i in range(3))
    emit("logsoftmax_stability", float_str(sum_exp_stable))

    # 2D shape preserved
    lx4 = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    ls4 = log_softmax(lx4, 1)
    emit("logsoftmax_shape", shape_str(ls4.shape()))

    # All logSoftmax values ≤ 0
    all_neg = all(ls1.data.get(i) <= 1e-6 for i in range(3))
    emit("logsoftmax_negative", str(all_neg).lower())

    # ===== CrossEntropy =====

    # Perfect prediction → loss ≈ 0
    ce_logits1 = Tensor.from_array([100, 0, 0], 1, 3)
    ce_targets1 = Tensor.from_array([1, 0, 0], 1, 3)
    ce1 = cross_entropy(ce_logits1, ce_targets1)
    emit("ce_perfect", float_str(ce1.data.get(0)))

    # Uniform logits → loss = log(3) ≈ 1.0986
    ce_logits2 = Tensor.from_array([1, 1, 1], 1, 3)
    ce_targets2 = Tensor.from_array([1, 0, 0], 1, 3)
    ce2 = cross_entropy(ce_logits2, ce_targets2)
    emit("ce_uniform", float_str(ce2.data.get(0)))

    # Known: logits=[2,1,0.1], targets=[1,0,0] → CE ≈ 0.4170
    ce_logits3 = Tensor.from_array([2, 1, 0.1], 1, 3)
    ce_targets3 = Tensor.from_array([1, 0, 0], 1, 3)
    ce3 = cross_entropy(ce_logits3, ce_targets3)
    emit("ce_known", float_str(ce3.data.get(0)))

    # Batch of 2: average over samples
    ce_logits4 = Tensor.from_array([2, 1, 0.1, 0.5, 2.5, 0.3], 2, 3)
    ce_targets4 = Tensor.from_array([1, 0, 0, 0, 1, 0], 2, 3)
    ce4 = cross_entropy(ce_logits4, ce_targets4)
    emit("ce_batch", float_str(ce4.data.get(0)))

    # Wrong prediction → high loss
    ce_logits5 = Tensor.from_array([0, 0, 2], 1, 3)
    ce_targets5 = Tensor.from_array([1, 0, 0], 1, 3)
    ce5 = cross_entropy(ce_logits5, ce_targets5)
    emit("ce_wrong", float_str(ce5.data.get(0)))

    # Shape: should be [1]
    emit("ce_shape", shape_str(ce3.shape()))


if __name__ == "__main__":
    main()
