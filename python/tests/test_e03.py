"""test_e03.py — E03 Linear Layer test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify your Layer implementations.
"""

import os
import sys

# Ensure packages are importable regardless of how this script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor
from tinytorch.layer import Layer, Linear, Dropout, Sequential


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:

    # ===== Linear basics =====

    lin = Linear(3, 2)
    emit("linear_weight_shape", shape_str(lin.weight.shape()))
    emit("linear_bias_shape", shape_str(lin.bias.shape()))
    emit("linear_bias_init", str(lin.bias))
    emit("linear_params_count", str(len(lin.parameters())))

    lin_nb = Linear(3, 2, bias=False)
    emit("linear_no_bias_params", str(len(lin_nb.parameters())))
    emit("linear_no_bias_null", str(lin_nb.bias is None).lower())

    # ===== Linear forward with known weights =====

    # W = [[1,0,0],[0,1,0]]  shape [2,3]
    # b = [10, 20]
    lin.weight = Tensor.from_array([1, 0, 0, 0, 1, 0], 2, 3)
    lin.bias = Tensor.from_array([10, 20], 2)

    # x = [[1,2,3],[4,5,6]]  shape [2,3]
    # y = x @ W.T + b = [[1,2],[4,5]] + [10,20] = [[11,22],[14,25]]
    x = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    y = lin.forward(x)
    emit("linear_forward_shape", shape_str(y.shape()))
    emit("linear_forward_toString", str(y))

    # Without bias
    lin_nb.weight = Tensor.from_array([1, 0, 0, 0, 1, 0], 2, 3)
    y_nb = lin_nb.forward(Tensor.from_array([1, 2, 3], 1, 3))
    emit("linear_forward_no_bias", str(y_nb))

    # ===== LeCun init variance check =====

    lin_big = Linear(1000, 50)
    w = lin_big.weight
    # Compute global mean (keepdim=True to avoid 0D scalar)
    w_mean = w.mean(1, True).mean(0, True).data.get(0, 0)
    # Compute global variance
    diff = w.sub(Tensor.full(w_mean, *w.shape()))
    diff_sq = diff.mul(diff)
    w_var = diff_sq.mean(1, True).mean(0, True).data.get(0, 0)
    emit("linear_init_variance", float_str(w_var))

    # ===== Dropout =====

    dx = Tensor.from_array([1, 2, 3, 4], 2, 2)

    # Eval mode: identity
    drop_eval = Dropout(0.5)
    drop_eval.eval()
    emit("dropout_eval", str(drop_eval.forward(dx)))

    # p=0 in train mode: identity
    drop0 = Dropout(0.0)
    emit("dropout_p0_train", str(drop0.forward(dx)))

    # p=1 in train mode: all zeros
    drop1 = Dropout(1.0)
    emit("dropout_p1_train", str(drop1.forward(dx)))

    # Shape preserved (using eval mode for determinism)
    emit("dropout_shape", shape_str(drop_eval.forward(dx).shape()))

    # No parameters
    emit("dropout_no_params", str(len(Dropout(0.5).parameters())))

    # ===== Sequential =====

    sl1 = Linear(3, 2)
    sl1.weight = Tensor.from_array([1, 0, 0, 0, 1, 0], 2, 3)
    sl1.bias = Tensor.from_array([0, 0], 2)

    sl2 = Linear(2, 1)
    sl2.weight = Tensor.from_array([1, 1], 1, 2)
    sl2.bias = Tensor.from_array([0], 1)

    seq = Sequential(sl1, sl2)

    # forward: [[1,2,3]] → sl1 → [[1,2]] → sl2 → [[3]]
    sx = Tensor.from_array([1, 2, 3], 1, 3)
    sy = seq.forward(sx)
    emit("sequential_forward_shape", shape_str(sy.shape()))
    emit("sequential_forward", str(sy))

    # Parameters: sl1(weight+bias) + sl2(weight+bias) = 4
    emit("sequential_params_count", str(len(seq.parameters())))

    # Children
    emit("sequential_children_count", str(len(seq.children())))

    # ===== Train / Eval mode =====

    emit("training_default", str(Linear(3, 2).training).lower())

    lin_mode = Linear(3, 2)
    lin_mode.eval()
    emit("eval_sets_false", str(lin_mode.training).lower())
    lin_mode.train()
    emit("train_sets_true", str(lin_mode.training).lower())

    # Recursive mode switching
    seq.eval()
    emit("eval_recursive", str(seq.children()[0].training).lower())
    seq.train()
    emit("train_recursive", str(seq.children()[0].training).lower())


if __name__ == "__main__":
    main()
