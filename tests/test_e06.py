"""test_e06.py — E06 More Backward Ops test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify backward implementations
for mat_mul, sum, mean, reshape, transpose, exp, log, activations, losses, and dropout.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch import activations as act
from tinytorch import losses
from tinytorch.function import DropoutBackward


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def shape_str(shape: tuple) -> str:
    return ",".join(str(d) for d in shape)


def main() -> None:

    # ===== Part 1: Tensor operations graph recording + backward =====

    # --- mat_mul ---
    a = Tensor.from_array([1, 2, 3, 4], 2, 2)
    a.requires_grad = True
    b = Tensor.from_array([5, 6, 7, 8], 2, 2)
    z_mat = a.mat_mul(b)
    emit("graph_matmul_has_fn", str(z_mat.grad_fn is not None).lower())
    emit("forward_matmul_00", float_str(z_mat.data.get(0, 0)))
    emit("forward_matmul_11", float_str(z_mat.data.get(1, 1)))
    grad_ones = NDArray.ones(2, 2)
    mat_grads = z_mat.grad_fn.backward(grad_ones)
    emit("backward_matmul_da_00", float_str(mat_grads[0].get(0, 0)))
    emit("backward_matmul_da_01", float_str(mat_grads[0].get(0, 1)))
    emit("backward_matmul_db_00", float_str(mat_grads[1].get(0, 0)))
    emit("backward_matmul_db_10", float_str(mat_grads[1].get(1, 0)))

    # --- sum ---
    s = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    s.requires_grad = True
    z_sum = s.sum(1, keep_dims=False)
    emit("graph_sum_has_fn", str(z_sum.grad_fn is not None).lower())
    emit("forward_sum_0", float_str(z_sum.data.get(0)))
    emit("forward_sum_1", float_str(z_sum.data.get(1)))
    sum_grads = z_sum.grad_fn.backward(NDArray.ones(2))
    emit("backward_sum_shape", shape_str(sum_grads[0].get_shape()))
    emit("backward_sum_00", float_str(sum_grads[0].get(0, 0)))

    # --- mean ---
    m = Tensor.from_array([2, 4, 6, 8], 2, 2)
    m.requires_grad = True
    z_mean = m.mean(1, keep_dims=False)
    emit("graph_mean_has_fn", str(z_mean.grad_fn is not None).lower())
    emit("forward_mean_0", float_str(z_mean.data.get(0)))
    mean_grads = z_mean.grad_fn.backward(NDArray.ones(2))
    emit("backward_mean_00", float_str(mean_grads[0].get(0, 0)))

    # --- reshape ---
    r = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    r.requires_grad = True
    z_reshape = r.reshape(3, 2)
    emit("graph_reshape_has_fn", str(z_reshape.grad_fn is not None).lower())
    emit("forward_reshape_shape", shape_str(z_reshape.data.get_shape()))
    reshape_grads = z_reshape.grad_fn.backward(NDArray.ones(3, 2))
    emit("backward_reshape_shape", shape_str(reshape_grads[0].get_shape()))

    # --- transpose ---
    t = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    t.requires_grad = True
    z_trans = t.transpose(1, 0)
    emit("graph_transpose_has_fn", str(z_trans.grad_fn is not None).lower())
    emit("forward_transpose_shape", shape_str(z_trans.data.get_shape()))
    emit("forward_transpose_00", float_str(z_trans.data.get(0, 0)))
    emit("forward_transpose_10", float_str(z_trans.data.get(1, 0)))
    trans_grads = z_trans.grad_fn.backward(NDArray.ones(3, 2))
    emit("backward_transpose_shape", shape_str(trans_grads[0].get_shape()))

    # --- exp ---
    e = Tensor.from_array([0.0, 1.0], 2)
    e.requires_grad = True
    z_exp = e.exp()
    emit("graph_exp_has_fn", str(z_exp.grad_fn is not None).lower())
    emit("forward_exp_0", float_str(z_exp.data.get(0)))
    emit("forward_exp_1", float_str(z_exp.data.get(1)))
    exp_grads = z_exp.grad_fn.backward(NDArray.ones(2))
    emit("backward_exp_0", float_str(exp_grads[0].get(0)))
    emit("backward_exp_1", float_str(exp_grads[0].get(1)))

    # --- log ---
    lx = Tensor.from_array([1.0, math.e], 2)
    lx.requires_grad = True
    z_log = lx.log()
    emit("graph_log_has_fn", str(z_log.grad_fn is not None).lower())
    emit("forward_log_0", float_str(z_log.data.get(0)))
    emit("forward_log_1", float_str(z_log.data.get(1)))
    log_grads = z_log.grad_fn.backward(NDArray.ones(2))
    emit("backward_log_0", float_str(log_grads[0].get(0)))
    emit("backward_log_1", float_str(log_grads[0].get(1)))

    # ===== Part 2: Activation backward =====

    ax = Tensor.from_array([-1.0, 0.0, 1.0, 2.0], 4)
    ax.requires_grad = True

    # --- relu ---
    z_relu = act.relu(ax)
    emit("graph_relu_has_fn", str(z_relu.grad_fn is not None).lower())
    emit("forward_relu_0", float_str(z_relu.data.get(0)))
    emit("forward_relu_2", float_str(z_relu.data.get(2)))
    relu_grads = z_relu.grad_fn.backward(NDArray.ones(4))
    emit("backward_relu_0", float_str(relu_grads[0].get(0)))
    emit("backward_relu_2", float_str(relu_grads[0].get(2)))

    # --- sigmoid ---
    z_sig = act.sigmoid(ax)
    emit("graph_sigmoid_has_fn", str(z_sig.grad_fn is not None).lower())
    emit("forward_sigmoid_1", float_str(z_sig.data.get(1)))
    sig_grads = z_sig.grad_fn.backward(NDArray.ones(4))
    emit("backward_sigmoid_1", float_str(sig_grads[0].get(1)))

    # --- tanh ---
    z_tanh = act.tanh(ax)
    emit("graph_tanh_has_fn", str(z_tanh.grad_fn is not None).lower())
    emit("forward_tanh_1", float_str(z_tanh.data.get(1)))
    tanh_grads = z_tanh.grad_fn.backward(NDArray.ones(4))
    emit("backward_tanh_1", float_str(tanh_grads[0].get(1)))

    # --- gelu ---
    z_gelu = act.gelu(ax)
    emit("graph_gelu_has_fn", str(z_gelu.grad_fn is not None).lower())
    emit("forward_gelu_1", float_str(z_gelu.data.get(1)))
    gelu_grads = z_gelu.grad_fn.backward(NDArray.ones(4))
    emit("backward_gelu_1", float_str(gelu_grads[0].get(1)))

    # ===== Part 3: Loss backward =====

    # --- cross_entropy ---
    logits = Tensor.from_array([2.0, 1.0, 0.1, 0.5, 2.5, 0.3], 2, 3)
    logits.requires_grad = True
    targets = Tensor.from_array([1, 0, 0, 0, 1, 0], 2, 3)
    ce = losses.cross_entropy(logits, targets)
    emit("graph_ce_has_fn", str(ce.grad_fn is not None).lower())
    emit("forward_ce_shape", shape_str(ce.data.get_shape()))
    ce_grads = ce.grad_fn.backward(NDArray.ones(1))
    emit("backward_ce_shape", shape_str(ce_grads[0].get_shape()))
    ce_grad_sum0 = ce_grads[0].get(0, 0) + ce_grads[0].get(0, 1) + ce_grads[0].get(0, 2)
    emit("backward_ce_row_sum", float_str(ce_grad_sum0))

    # --- mse (via Tensor ops) ---
    pred = Tensor.from_array([1.0, 2.0, 3.0], 3)
    pred.requires_grad = True
    target2 = Tensor.from_array([1.5, 2.5, 3.5], 3)
    mse_val = losses.mse(pred, target2)
    emit("forward_mse_shape", shape_str(mse_val.data.get_shape()))
    emit("graph_mse_has_fn", str(mse_val.grad_fn is not None).lower())

    # ===== Part 4: Dropout backward =====

    dx = Tensor.from_array([1, 2, 3, 4], 2, 2)
    dx.requires_grad = True
    mask = NDArray.from_array([2.0, 0.0, 2.0, 0.0], 2, 2)
    drop_fn = DropoutBackward(mask)
    drop_out = drop_fn.forward(dx)[0]
    emit("forward_dropout_0", float_str(drop_out.data.get(0, 0)))
    emit("forward_dropout_1", float_str(drop_out.data.get(0, 1)))
    drop_grads = drop_fn.backward(NDArray.ones(2, 2))
    emit("backward_dropout_0", float_str(drop_grads[0].get(0, 0)))
    emit("backward_dropout_1", float_str(drop_grads[0].get(0, 1)))


if __name__ == "__main__":
    main()
