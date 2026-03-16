"""test_e01.py — E01 Tensor Class test driver.

Provided by tinytorch-starter. Do NOT modify.
The tester runs this file to verify your Tensor implementation.
"""

import os
import sys

# Ensure packages are importable regardless of how this script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def main() -> None:
    # --- zeros ---
    z = Tensor.zeros(2, 3)
    emit("zeros_size", str(z.size()))
    emit("zeros_ndim", str(z.ndim()))
    emit("zeros_shape", shape_str(z.shape()))
    emit("zeros_toString", str(z))

    # --- ones ---
    o = Tensor.ones(3, 4)
    emit("ones_size", str(o.size()))
    emit("ones_shape", shape_str(o.shape()))
    emit("ones_toString", str(Tensor.ones(2, 3)))

    # --- from_array 1D ---
    a1d = Tensor.from_array([1, 2, 3], 3)
    emit("fromArray_1d_toString", str(a1d))

    # --- from_array 2D ---
    a = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("fromArray_2d_toString", str(a))

    # --- full ---
    f = Tensor.full(7.0, 2, 2)
    emit("full_toString", str(f))

    # --- add ---
    x = Tensor.from_array([1, 2, 3, 4], 2, 2)
    y = Tensor.ones(2, 2)
    emit("add_toString", str(x.add(y)))

    # --- sub ---
    emit("sub_toString", str(x.sub(y)))

    # --- mul ---
    emit("mul_toString", str(x.mul(y)))

    # --- div ---
    d = Tensor.from_array([2, 4, 6, 8], 2, 2)
    two = Tensor.full(2.0, 2, 2)
    emit("div_toString", str(d.div(two)))

    # --- mat_mul ---
    m1 = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    m2 = Tensor.from_array([1, 2, 3, 4, 5, 6], 3, 2)
    prod = m1.mat_mul(m2)
    emit("matMul_shape", shape_str(prod.shape()))
    emit("matMul_toString", str(prod))

    # --- sum ---
    s = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("sum_axis0", str(s.sum(0, False)))
    emit("sum_axis1", str(s.sum(1, False)))

    # --- mean ---
    emit("mean_axis0_keepDims", str(s.mean(0, True)))
    emit("mean_axis1", str(s.mean(1, False)))

    # --- reshape ---
    r = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("reshape_shape", shape_str(r.reshape(3, 2).shape()))
    emit("reshape_toString", str(r.reshape(3, 2)))

    # --- transpose ---
    t = Tensor.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("transpose_shape", shape_str(t.transpose(1, 0).shape()))
    emit("transpose_toString", str(t.transpose(1, 0)))

    # --- gradient fields (dormant) ---
    g = Tensor.zeros(3, 3)
    emit("requiresGrad", str(g.requires_grad).lower())
    emit("grad_null", str(g.grad is None).lower())
    emit("gradFn_null", str(g.grad_fn is None).lower())

    # --- randn shape ---
    rn = Tensor.randn(4, 5)
    emit("randn_shape", shape_str(rn.shape()))
    emit("randn_size", str(rn.size()))


if __name__ == "__main__":
    main()
