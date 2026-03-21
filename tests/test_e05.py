"""test_e05.py — E05 Computation Graph test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify your backward classes
and computation graph recording in Tensor operations.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor
from tinytorch.function import Function


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:

    # Shared inputs: x requires grad, y does not
    x = Tensor.from_array([2.0, 3.0], 2)
    x.requires_grad = True
    y = Tensor.from_array([4.0, 5.0], 2)

    # Gradient flowing back from downstream
    grad_out = Tensor.ones(2).data

    # ===== Graph recording =====

    z_add = x.add(y)
    emit("graph_add_has_fn", str(z_add.grad_fn is not None).lower())

    z_sub = x.sub(y)
    emit("graph_sub_has_fn", str(z_sub.grad_fn is not None).lower())

    z_mul = x.mul(y)
    emit("graph_mul_has_fn", str(z_mul.grad_fn is not None).lower())

    z_div = x.div(y)
    emit("graph_div_has_fn", str(z_div.grad_fn is not None).lower())

    # No grad: both inputs have requires_grad=False
    a = Tensor.from_array([1.0, 2.0], 2)
    b = Tensor.from_array([3.0, 4.0], 2)
    no_grad = a.add(b)
    emit("graph_no_grad", str(no_grad.grad_fn is None).lower())

    # requires_grad propagation
    emit("graph_requires_grad", str(z_add.requires_grad).lower())

    # get_inputs() returns saved inputs
    emit("graph_inputs_count", str(len(z_add.grad_fn.get_inputs())))

    # ===== Forward values (graph path should produce correct results) =====

    emit("forward_add_0", float_str(z_add.data.get(0)))
    emit("forward_add_1", float_str(z_add.data.get(1)))
    emit("forward_sub_0", float_str(z_sub.data.get(0)))
    emit("forward_mul_0", float_str(z_mul.data.get(0)))
    emit("forward_div_0", float_str(z_div.data.get(0)))

    # ===== Backward gradients =====

    # AddBackward: grad_a = grad_out, grad_b = grad_out
    add_grads = z_add.grad_fn.backward(grad_out)
    emit("backward_add_da", float_str(add_grads[0].get(0)))
    emit("backward_add_db", float_str(add_grads[1].get(0)))

    # SubBackward: grad_a = grad_out, grad_b = -grad_out
    sub_grads = z_sub.grad_fn.backward(grad_out)
    emit("backward_sub_da", float_str(sub_grads[0].get(0)))
    emit("backward_sub_db", float_str(sub_grads[1].get(0)))

    # MulBackward: grad_a = grad_out * y, grad_b = grad_out * x
    mul_grads = z_mul.grad_fn.backward(grad_out)
    emit("backward_mul_da", float_str(mul_grads[0].get(0)))
    emit("backward_mul_db", float_str(mul_grads[1].get(0)))

    # DivBackward: grad_a = grad_out / y, grad_b = -grad_out * x / y²
    div_grads = z_div.grad_fn.backward(grad_out)
    emit("backward_div_da", float_str(div_grads[0].get(0)))
    emit("backward_div_db", float_str(div_grads[1].get(0)))


if __name__ == "__main__":
    main()
