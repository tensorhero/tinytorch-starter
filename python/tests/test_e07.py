"""test_e07.py — E07 Backpropagation test driver.

Provided by tinytorch-starter. Do NOT modify.
Tests: backward(), topological_sort(), reduce_broadcast_grad(), no_grad().
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch import activations as act
from tinytorch.tensor import topological_sort


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def shape_str(shape: tuple) -> str:
    return ",".join(str(d) for d in shape)


def main() -> None:

    # ===== Part 1: Topological Sort =====

    x = Tensor.from_array([1, 2, 3, 4], 2, 2)
    x.requires_grad = True
    w = Tensor.from_array([0.5, -0.5, 0.5, -0.5], 2, 2)
    w.requires_grad = True
    y = x.mat_mul(w)  # y = x @ w
    order = topological_sort(y)
    emit("topo_size", str(len(order)))
    emit("topo_root_last", str(order[-1] is y).lower())
    x_idx = order.index(x)
    w_idx = order.index(w)
    y_idx = order.index(y)
    emit("topo_x_before_y", str(x_idx < y_idx).lower())
    emit("topo_w_before_y", str(w_idx < y_idx).lower())

    # ===== Part 2: Simple backward (y = a * b, scalar) =====

    a = Tensor.from_array([3.0], 1)
    a.requires_grad = True
    b = Tensor.from_array([4.0], 1)
    b.requires_grad = True
    c = a.mul(b)  # c = a * b = 12
    c.backward()
    emit("simple_backward_a_grad", float_str(a.grad.data.get(0)))
    emit("simple_backward_b_grad", float_str(b.grad.data.get(0)))

    # ===== Part 3: Multi-op backward =====

    x2 = Tensor.from_array([1, 2, 3, 4], 2, 2)
    x2.requires_grad = True
    w2 = Tensor.from_array([0.5, -0.5, 0.5, -0.5], 2, 2)
    w2.requires_grad = True
    b2 = Tensor.from_array([0.1, -0.1], 2)
    b2.requires_grad = True

    mm = x2.mat_mul(w2)
    added = mm.add(b2)
    activated = act.relu(added)
    summed = activated.sum(1, keep_dims=False)
    loss = summed.mean(0, keep_dims=True)

    loss.backward()

    emit("multi_x_grad_exists", str(x2.grad is not None).lower())
    emit("multi_w_grad_exists", str(w2.grad is not None).lower())
    emit("multi_b_grad_exists", str(b2.grad is not None).lower())

    emit("multi_x_grad_shape", shape_str(x2.grad.data.get_shape()))
    emit("multi_w_grad_shape", shape_str(w2.grad.data.get_shape()))
    emit("multi_b_grad_shape", shape_str(b2.grad.data.get_shape()))

    emit("multi_loss_value", float_str(loss.data.get(0)))

    emit("multi_b_grad_0", float_str(b2.grad.data.get(0)))
    emit("multi_b_grad_1", float_str(b2.grad.data.get(1)))

    # ===== Part 4: Broadcast gradient reduction =====

    x3 = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    x3.requires_grad = True
    b3 = Tensor.from_array([0.1, 0.2, 0.3], 3)
    b3.requires_grad = True
    sum3 = x3.add(b3).sum(1, keep_dims=False).mean(0, keep_dims=True)
    sum3.backward()

    emit("broadcast_b_grad_shape", shape_str(b3.grad.data.get_shape()))
    emit("broadcast_b_grad_0", float_str(b3.grad.data.get(0)))
    emit("broadcast_b_grad_1", float_str(b3.grad.data.get(1)))
    emit("broadcast_b_grad_2", float_str(b3.grad.data.get(2)))

    # ===== Part 5: Gradient accumulation (x used twice: z = x + x) =====

    x4 = Tensor.from_array([1.0, 2.0, 3.0], 3)
    x4.requires_grad = True
    z4 = x4.add(x4)
    s4 = z4.sum(0, keep_dims=True)
    s4.backward()
    emit("accum_grad_0", float_str(x4.grad.data.get(0)))
    emit("accum_grad_1", float_str(x4.grad.data.get(1)))
    emit("accum_grad_2", float_str(x4.grad.data.get(2)))

    # ===== Part 6: noGrad =====

    x5 = Tensor.from_array([1.0, 2.0], 2)
    x5.requires_grad = True
    y5 = Tensor.from_array([3.0, 4.0], 2)

    result = Tensor.no_grad(lambda: x5.add(y5))
    emit("nograd_result_0", float_str(result.data.get(0)))
    emit("nograd_result_1", float_str(result.data.get(1)))
    emit("nograd_no_fn", str(result.grad_fn is None).lower())

    # Verify grad_enabled is restored
    z5 = x5.add(y5)
    emit("nograd_restored", str(z5.grad_fn is not None).lower())

    # ===== Part 7: Finite difference verification =====

    x_fd = Tensor.from_array([1.0, 2.0, 3.0], 3)
    x_fd.requires_grad = True
    sqr = x_fd.mul(x_fd)
    s_fd = sqr.sum(0, keep_dims=True)
    s_fd.backward()
    emit("fd_grad_0", float_str(x_fd.grad.data.get(0)))
    emit("fd_grad_1", float_str(x_fd.grad.data.get(1)))
    emit("fd_grad_2", float_str(x_fd.grad.data.get(2)))


if __name__ == "__main__":
    main()
