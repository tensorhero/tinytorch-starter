"""test_e02.py — E02 Activations test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify your Activations implementation.
"""

import os
import sys

# Ensure packages are importable regardless of how this script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor, activations


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def get_first(t: Tensor) -> float:
    """Get the first element of a 1D tensor."""
    return t.data.get(0)


def get_at(t: Tensor, i: int) -> float:
    """Get element at index i of a 1D tensor."""
    return t.data.get(i)


def get_at_2d(t: Tensor, row: int, col: int) -> float:
    """Get element at (row, col) of a 2D tensor."""
    return t.data.get(row, col)


def main() -> None:

    # ===== ReLU =====

    # relu basic: [-1, 0, 1, 2] → [0, 0, 1, 2]
    rx = Tensor.from_array([-1, 0, 1, 2], 4)
    ry = activations.relu(rx)
    emit("relu_basic", str(ry))

    # relu all negative: [-3, -2, -1] → [0, 0, 0]
    rn = Tensor.from_array([-3, -2, -1], 3)
    emit("relu_all_negative", str(activations.relu(rn)))

    # relu preserves shape
    r2d = Tensor.from_array([-1, 2, -3, 4], 2, 2)
    emit("relu_shape", shape_str(activations.relu(r2d).shape()))

    # ===== Sigmoid =====

    # sigmoid(0) = 0.5
    sz = Tensor.from_array([0], 1)
    emit("sigmoid_zero", float_str(get_first(activations.sigmoid(sz))))

    # sigmoid basic: [-2, 0, 2]
    sx = Tensor.from_array([-2, 0, 2], 3)
    sy = activations.sigmoid(sx)
    emit("sigmoid_neg2", float_str(get_at(sy, 0)))
    emit("sigmoid_pos2", float_str(get_at(sy, 2)))

    # sigmoid symmetry: sigmoid(-x) + sigmoid(x) ≈ 1.0
    s5 = Tensor.from_array([5], 1)
    sn5 = Tensor.from_array([-5], 1)
    sig_sum = get_first(activations.sigmoid(s5)) + get_first(activations.sigmoid(sn5))
    emit("sigmoid_symmetry_sum", float_str(sig_sum))

    # ===== Tanh =====

    # tanh(0) = 0.0
    tz = Tensor.from_array([0], 1)
    emit("tanh_zero", float_str(get_first(activations.tanh(tz))))

    # tanh basic: [0, 1, -1]
    tx = Tensor.from_array([0, 1, -1], 3)
    ty = activations.tanh(tx)
    emit("tanh_pos1", float_str(get_at(ty, 1)))
    emit("tanh_neg1", float_str(get_at(ty, 2)))

    # tanh antisymmetry: tanh(-x) = -tanh(x) → sum ≈ 0
    t3 = Tensor.from_array([3], 1)
    tn3 = Tensor.from_array([-3], 1)
    tanh_sum = get_first(activations.tanh(t3)) + get_first(activations.tanh(tn3))
    emit("tanh_antisymmetry_sum", float_str(tanh_sum))

    # ===== GELU =====

    # gelu(0) = 0.0
    gz = Tensor.from_array([0], 1)
    emit("gelu_zero", float_str(get_first(activations.gelu(gz))))

    # gelu basic: [-1, 0, 1]
    gx = Tensor.from_array([-1, 0, 1], 3)
    gy = activations.gelu(gx)
    emit("gelu_neg1", float_str(get_at(gy, 0)))
    emit("gelu_pos1", float_str(get_at(gy, 2)))

    # gelu preserves large positive values approximately
    g3 = Tensor.from_array([3], 1)
    emit("gelu_pos3", float_str(get_first(activations.gelu(g3))))

    # ===== Softmax =====

    # softmax basic: [1, 2, 3] along axis=0
    smx = Tensor.from_array([1, 2, 3], 3)
    smy = activations.softmax(smx, 0)
    emit("softmax_val0", float_str(get_at(smy, 0)))
    emit("softmax_val1", float_str(get_at(smy, 1)))
    emit("softmax_val2", float_str(get_at(smy, 2)))

    # softmax sum ≈ 1.0
    sm_sum = get_at(smy, 0) + get_at(smy, 1) + get_at(smy, 2)
    emit("softmax_sum", float_str(sm_sum))

    # softmax uniform: [1, 1, 1] → [1/3, 1/3, 1/3]
    smu = Tensor.from_array([1, 1, 1], 3)
    smuy = activations.softmax(smu, 0)
    emit("softmax_uniform_val", float_str(get_at(smuy, 0)))

    # softmax numerical stability: [1000, 1001, 1002] should not overflow
    sml = Tensor.from_array([1000, 1001, 1002], 3)
    smly = activations.softmax(sml, 0)
    sml_sum = get_at(smly, 0) + get_at(smly, 1) + get_at(smly, 2)
    emit("softmax_stability_sum", float_str(sml_sum))
    emit("softmax_stability_val2", float_str(get_at(smly, 2)))

    # softmax 2D: [[1,2,3],[1,1,1]] along axis=1
    sm2d = Tensor.from_array([1, 2, 3, 1, 1, 1], 2, 3)
    sm2dy = activations.softmax(sm2d, 1)
    emit("softmax_2d_shape", shape_str(sm2dy.shape()))

    # row 0 sum ≈ 1.0
    row0sum = get_at_2d(sm2dy, 0, 0) + get_at_2d(sm2dy, 0, 1) + get_at_2d(sm2dy, 0, 2)
    emit("softmax_2d_row0_sum", float_str(row0sum))

    # row 1 should be uniform (all ≈ 1/3)
    emit("softmax_2d_row1_val", float_str(get_at_2d(sm2dy, 1, 0)))


if __name__ == "__main__":
    main()
