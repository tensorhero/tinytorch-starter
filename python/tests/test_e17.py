"""test_e17.py — E17 Profiling & Compression test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.layer import Linear, Sequential
from tinytorch.profiler import count_params, count_flops
from tinytorch.pruner import magnitude_prune, measure_sparsity


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    # ============================================================
    # Part 1: Profiler — count_params
    # ============================================================

    # 1. Single Linear(4, 3) with bias: 4*3 + 3 = 15 params
    lin1 = Linear(4, 3)
    params1 = count_params(lin1)
    emit("count_params_single_linear", str(params1))

    # 2. Linear without bias: 4*3 = 12 params
    lin2 = Linear(4, 3, bias=False)
    params2 = count_params(lin2)
    emit("count_params_no_bias", str(params2))

    # 3. Sequential(Linear(4,8), Linear(8,2)): (4*8+8) + (8*2+2) = 58
    seq = Sequential(Linear(4, 8), Linear(8, 2))
    params3 = count_params(seq)
    emit("count_params_sequential", str(params3))

    # ============================================================
    # Part 1: Profiler — count_flops
    # ============================================================

    # 4. Linear(4,3) FLOPs: 2 * 4 * 3 = 24
    lin3 = Linear(4, 3)
    flops1 = count_flops(lin3, (1, 4))
    emit("count_flops_single_linear", str(flops1))

    # 5. Sequential(Linear(128,64), Linear(64,10)):
    #    2*128*64 + 2*64*10 = 16384 + 1280 = 17664
    seq2 = Sequential(Linear(128, 64), Linear(64, 10))
    flops2 = count_flops(seq2, (1, 128))
    emit("count_flops_sequential", str(flops2))

    # 6. Empty model (no Linear layers): 0 FLOPs
    empty_seq = Sequential()
    flops3 = count_flops(empty_seq, (1, 4))
    emit("count_flops_empty", str(flops3))

    # ============================================================
    # Part 2: Pruner — magnitude_prune
    # ============================================================

    # 7. Prune with known values: [0.1, -0.8, 0.3, -0.05, 0.9, -0.2]
    #    abs = [0.1, 0.8, 0.3, 0.05, 0.9, 0.2]
    #    50% percentile of abs → threshold ≈ 0.25
    #    Values with |x| < 0.25: indices 0 (0.1), 3 (0.05), 5 (0.2)
    #    After prune: [0, -0.8, 0.3, 0, 0.9, 0]
    w1 = Tensor.from_array(
        [0.1, -0.8, 0.3, -0.05, 0.9, -0.2], 2, 3)
    magnitude_prune(w1, 0.5)
    # Count non-zeros: should be 3
    non_zeros = 0
    for r in range(2):
        for c in range(3):
            if abs(w1.data.get(r, c)) > 1e-6:
                non_zeros += 1
    emit("prune_nonzero_count", str(non_zeros))

    # 8. The large values should survive: -0.8 at (0,1) should be non-zero
    big_survived = abs(w1.data.get(0, 1) - (-0.8)) < 1e-4
    emit("prune_big_value_survives", str(big_survived).lower())

    # 9. The small values should be zero: 0.1 at (0,0) should be zero
    small_pruned = abs(w1.data.get(0, 0)) < 1e-6
    emit("prune_small_value_zero", str(small_pruned).lower())

    # 10. Prune with sparsity=0 — nothing pruned
    w2 = Tensor.from_array([1.0, 2.0, 3.0, 4.0], 2, 2)
    magnitude_prune(w2, 0.0)
    sum2 = w2.data.flatten().sum(0, keep_dims=True).get(0)
    no_prune = abs(sum2 - 10.0) < 1e-4
    emit("prune_zero_sparsity", str(no_prune).lower())

    # ============================================================
    # Part 2: Pruner — measure_sparsity
    # ============================================================

    # 11. Model with no zeros: sparsity ≈ 0%
    lin_a = Linear(4, 3, bias=False)
    lin_a.weight = Tensor.from_ndarray(NDArray.ones(3, 4))
    sp1 = measure_sparsity(lin_a)
    sp_zero = sp1 < 0.01
    emit("sparsity_all_ones", str(sp_zero).lower())

    # 12. Model with known zeros: 4 out of 12 = 33.33%
    lin_b = Linear(4, 3, bias=False)
    lin_b.weight = Tensor.from_array([
        1.0, 0.0, 2.0, 0.0,
        3.0, 4.0, 0.0, 5.0,
        6.0, 7.0, 8.0, 0.0,
    ], 3, 4)
    sp2 = measure_sparsity(lin_b)
    # 4/12 * 100 = 33.333...
    emit("sparsity_known_zeros", float_str(sp2))

    # 13. Bias is NOT counted in sparsity (1D parameter)
    lin_c = Linear(2, 2)
    lin_c.weight = Tensor.from_ndarray(NDArray.ones(2, 2))
    lin_c.bias = Tensor.from_ndarray(NDArray.zeros(2))  # all-zero bias
    sp3 = measure_sparsity(lin_c)
    # weight has 0 zeros out of 4 → 0%
    # bias is 1D → skipped
    bias_skipped = sp3 < 0.01
    emit("sparsity_ignores_bias", str(bias_skipped).lower())

    # 14. Prune then measure: roundtrip consistency
    lin_d = Linear(8, 4, bias=False)
    # Set weight to known values: arange 0..31 reshaped (4, 8)
    #   has one zero at position (0,0)
    lin_d.weight = Tensor.from_ndarray(NDArray.arange(0, 32, 1).reshape(4, 8))
    magnitude_prune(lin_d.weight, 0.3)
    sp4 = measure_sparsity(lin_d)
    # After pruning 30% of 32 values → ~10 zeros
    # Sparsity should be >= 25% (at least close to 30%)
    sp_after_prune = sp4 >= 25.0
    emit("sparsity_after_prune", str(sp_after_prune).lower())


if __name__ == "__main__":
    main()
