"""test_e14.py — E14 Transformer Block test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.transformer_block import LayerNorm, MLP, Block


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    # ============================================================
    # Part 1: LayerNorm
    # ============================================================

    ln = LayerNorm(4)

    # x = [[1,2,3,4]], shape [1,4]
    ln_input = Tensor.from_array([1, 2, 3, 4], 1, 4)
    ln_out = ln.forward(ln_input)

    # 1. Output shape preserved
    emit("ln_output_shape", shape_str(ln_out.data.get_shape()))

    # 2. Specific output value: normalized[0,0] = (1-2.5)/sqrt(1.25+eps) ≈ -1.3416
    emit("ln_output_value_00", float_str(ln_out.data.get(0, 0)))

    # 3. Specific output value: normalized[0,3] = (4-2.5)/sqrt(1.25+eps) ≈ 1.3416
    emit("ln_output_value_03", float_str(ln_out.data.get(0, 3)))

    # 4. Mean of output row ≈ 0 (beta=0)
    row_mean = (
        ln_out.data.get(0, 0)
        + ln_out.data.get(0, 1)
        + ln_out.data.get(0, 2)
        + ln_out.data.get(0, 3)
    ) / 4.0
    emit("ln_output_mean", float_str(row_mean))

    # 5. Parameters count
    emit("ln_params_count", str(len(ln.parameters())))

    # ============================================================
    # Part 1b: LayerNorm Backward
    # ============================================================

    ln_bwd = LayerNorm(4)
    ln_bwd_input = Tensor.from_array([1, 2, 3, 4], 1, 4)
    ln_bwd_input.requires_grad = True
    ln_bwd_out = ln_bwd.forward(ln_bwd_input)
    # loss = sum of all output elements
    ln_loss = ln_bwd_out.sum(1, False).sum(0, False)
    ln_loss.backward()

    # 6. beta.grad[0] = 1.0 (one sample, upstream grad = 1)
    emit("ln_backward_beta_grad_0", float_str(ln_bwd.beta.grad.data.get(0)))

    # 7. gamma.grad[0] ≈ -1.3416 (= normalized[0,0])
    emit("ln_backward_gamma_grad_0", float_str(ln_bwd.gamma.grad.data.get(0)))

    # ============================================================
    # Part 1c: LayerNorm — uniform input (edge case)
    # ============================================================

    ln_unif = LayerNorm(4)
    unif_input = Tensor.from_array([5, 5, 5, 5], 1, 4)
    unif_out = ln_unif.forward(unif_input)

    # 8. Uniform input: diff=0, normalized=0 → output = 0*gamma+beta = beta = 0
    emit("ln_uniform_output_00", float_str(unif_out.data.get(0, 0)))

    # ============================================================
    # Part 2: MLP
    # ============================================================

    mlp = MLP(8)
    mlp_input = Tensor.ones(1, 3, 8)
    mlp_out = mlp.forward(mlp_input)

    # 9. Output shape preserved
    emit("mlp_output_shape", shape_str(mlp_out.data.get_shape()))

    # 10. Parameters count: fc1(weight+bias) + fc2(weight+bias) = 4
    emit("mlp_params_count", str(len(mlp.parameters())))

    # 11. Children count: fc1, fc2
    emit("mlp_children_count", str(len(mlp.children())))

    # ============================================================
    # Part 3: TransformerBlock
    # ============================================================

    block = Block(8, 2)
    block_input = Tensor.ones(1, 3, 8)
    block_out = block.forward(block_input)

    # 12. Output shape preserved (residual connection)
    emit("block_output_shape", shape_str(block_out.data.get_shape()))

    # 13. Parameters count: ln1(2) + attn(8) + ln2(2) + mlp(4) = 16
    emit("block_params_count", str(len(block.parameters())))

    # 14. Children count: ln1, attn, ln2, mlp = 4
    emit("block_children_count", str(len(block.children())))

    # ============================================================
    # Part 3b: Residual connection verification
    # ============================================================

    # 15. With all-zeros input: shape preserved through residual
    zero_input = Tensor.zeros(1, 3, 8)
    zero_out = block.forward(zero_input)
    emit("block_residual_shape", shape_str(zero_out.data.get_shape()))


if __name__ == "__main__":
    main()
