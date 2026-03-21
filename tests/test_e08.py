"""test_e08.py — E08 Optimizers test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinytorch import Tensor
from tinytorch.optimizer import SGD, Adam, AdamW


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def get_at(t: Tensor, i: int) -> float:
    return t.data.get(i)


def main() -> None:
    # =============================================================
    # Test 1–2: SGD basic step
    # param = [2.0, 4.0], grad = [1.0, 2.0], lr = 0.5
    # After step: param = [1.5, 3.0]
    # =============================================================
    p = Tensor.from_array([2.0, 4.0], 2)
    p.requires_grad = True
    p.grad = Tensor.from_array([1.0, 2.0], 2)

    sgd = SGD([p], lr=0.5)
    sgd.step()

    emit("sgd_step_param0", float_str(get_at(p, 0)))
    emit("sgd_step_param1", float_str(get_at(p, 1)))

    # =============================================================
    # Test 3: SGD zeroGrad
    # =============================================================
    p = Tensor.from_array([1.0], 1)
    p.requires_grad = True
    p.grad = Tensor.from_array([0.5], 1)

    sgd = SGD([p], lr=0.1)
    sgd.zero_grad()

    emit("sgd_zero_grad", str(p.grad is None).lower())

    # =============================================================
    # Test 4–5: SGD with momentum (two steps)
    # param = [1.0], lr = 0.1, momentum = 0.9
    # Step 1: grad=[2.0], v = 2.0, param = 1.0 - 0.1*2.0 = 0.8
    # Step 2: grad=[2.0], v = 0.9*2.0 + 2.0 = 3.8, param = 0.8 - 0.1*3.8 = 0.42
    # =============================================================
    p = Tensor.from_array([1.0], 1)
    p.requires_grad = True
    p.grad = Tensor.from_array([2.0], 1)

    sgd = SGD([p], lr=0.1, momentum=0.9)
    sgd.step()
    emit("sgd_momentum_step1", float_str(get_at(p, 0)))

    p.grad = Tensor.from_array([2.0], 1)
    sgd.step()
    emit("sgd_momentum_step2", float_str(get_at(p, 0)))

    # =============================================================
    # Test 6: SGD skips params without gradient
    # =============================================================
    p1 = Tensor.from_array([5.0], 1)
    p1.requires_grad = True
    # p1.grad is None — should be skipped

    p2 = Tensor.from_array([3.0], 1)
    p2.requires_grad = True
    p2.grad = Tensor.from_array([1.0], 1)

    sgd = SGD([p1, p2], lr=0.5)
    sgd.step()

    emit("sgd_no_grad_skip", str(get_at(p1, 0) == 5.0).lower())

    # =============================================================
    # Test 7–8: Adam basic (two steps)
    # param = [3.0], grad = [1.0], lr = 0.01
    # Step 1: m_hat=1.0, v_hat=1.0, param ≈ 3.0 - 0.01 = 2.99
    # Step 2: m_hat=1.0, v_hat=1.0, param ≈ 2.99 - 0.01 = 2.98
    # =============================================================
    p = Tensor.from_array([3.0], 1)
    p.requires_grad = True
    p.grad = Tensor.from_array([1.0], 1)

    adam = Adam([p], lr=0.01)
    adam.step()
    emit("adam_step1", float_str(get_at(p, 0)))

    p.grad = Tensor.from_array([1.0], 1)
    adam.step()
    emit("adam_step2", float_str(get_at(p, 0)))

    # =============================================================
    # Test 9: AdamW weight decay
    # param = [2.0], grad = [1.0], lr = 0.01, weight_decay = 0.5
    # Weight decay: param = 2.0 * (1 - 0.01*0.5) = 1.99
    # Adam update: param = 1.99 - 0.01 ≈ 1.98
    # =============================================================
    p = Tensor.from_array([2.0], 1)
    p.requires_grad = True
    p.grad = Tensor.from_array([1.0], 1)

    adamw = AdamW([p], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.5)
    adamw.step()
    emit("adamw_step1", float_str(get_at(p, 0)))

    # =============================================================
    # Test 10: AdamW result < Adam result (weight decay effect)
    # =============================================================
    p_adam = Tensor.from_array([2.0], 1)
    p_adam.requires_grad = True
    p_adam.grad = Tensor.from_array([1.0], 1)
    adam = Adam([p_adam], lr=0.01)
    adam.step()

    p_adamw = Tensor.from_array([2.0], 1)
    p_adamw.requires_grad = True
    p_adamw.grad = Tensor.from_array([1.0], 1)
    adamw = AdamW([p_adamw], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.5)
    adamw.step()

    emit("adamw_smaller_than_adam", str(get_at(p_adamw, 0) < get_at(p_adam, 0)).lower())

    # =============================================================
    # Test 11–13: Multiple parameters
    # p1 = [1.0, 2.0], grad = [0.5, 0.5]
    # p2 = [3.0], grad = [1.0]
    # SGD lr = 0.2
    # After step: p1 = [0.9, 1.9], p2 = [2.8]
    # =============================================================
    p1 = Tensor.from_array([1.0, 2.0], 2)
    p1.requires_grad = True
    p1.grad = Tensor.from_array([0.5, 0.5], 2)

    p2 = Tensor.from_array([3.0], 1)
    p2.requires_grad = True
    p2.grad = Tensor.from_array([1.0], 1)

    sgd = SGD([p1, p2], lr=0.2)
    sgd.step()

    emit("multi_params_p1_0", float_str(get_at(p1, 0)))
    emit("multi_params_p1_1", float_str(get_at(p1, 1)))
    emit("multi_params_p2_0", float_str(get_at(p2, 0)))


if __name__ == "__main__":
    main()
