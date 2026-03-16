"""test_e12.py — E12 Embeddings test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.embedding import Embedding, PositionalEncoding


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def get_at_2d(t: Tensor, row: int, col: int) -> float:
    return t.data.get(row, col)


def main() -> None:
    # ============================================================
    # Part 1: Embedding — lookup & basic properties
    # ============================================================

    emb = Embedding(4, 3)
    emb.weight = Tensor.from_array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4, 3
    )
    emb.weight.requires_grad = True

    # 1. Shape of forward output
    out = emb.forward([0, 2, 3])
    emit("embedding_shape", shape_str(out.data.get_shape()))

    # 2. Lookup correctness — row 2 element [0,0]
    out2 = emb.forward([2])
    emit("embedding_lookup_r0c0", float_str(get_at_2d(out2, 0, 0)))

    # 3. Lookup correctness — row 2 element [0,2]
    emit("embedding_lookup_r0c2", float_str(get_at_2d(out2, 0, 2)))

    # 4. Repeated index — both rows should be identical
    out_rep = emb.forward([1, 1])
    emit("embedding_repeated", float_str(get_at_2d(out_rep, 0, 0)))

    # 5. Parameters count
    emit("embedding_params_count", str(len(emb.parameters())))

    # 6. Weight shape
    emit("embedding_weight_shape", shape_str(emb.weight.data.get_shape()))

    # ============================================================
    # Part 2: Embedding — backward & gradient accumulation
    # ============================================================

    emb2 = Embedding(4, 3)
    emb2.weight = Tensor.from_array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4, 3
    )
    emb2.weight.requires_grad = True

    fwd = emb2.forward([2])
    loss = fwd.sum(1, False).sum(0, False)
    loss.backward()

    # 7. Gradient shape
    emit("embedding_grad_shape", shape_str(emb2.weight.data.get_shape()))

    # 8. Gradient at looked-up row [2,0] = 1.0
    emit("embedding_grad_single", float_str(emb2.weight.grad.data.get(2, 0)))

    # 9. Gradient at non-looked-up row [0,0] = 0.0
    emit("embedding_grad_zero", float_str(emb2.weight.grad.data.get(0, 0)))

    # 10. Gradient accumulation with repeated index
    emb3 = Embedding(4, 3)
    emb3.weight = Tensor.from_array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4, 3
    )
    emb3.weight.requires_grad = True

    fwd3 = emb3.forward([1, 1])
    loss3 = fwd3.sum(1, False).sum(0, False)
    loss3.backward()
    emit("embedding_grad_accumulate", float_str(emb3.weight.grad.data.get(1, 0)))

    # ============================================================
    # Part 3: Sinusoidal Positional Encoding
    # ============================================================

    sin_pe = PositionalEncoding("sinusoidal", 100, 8)

    # 11. Output shape
    inp = Tensor.ones(5, 8)
    sin_out = sin_pe.forward(inp)
    emit("sinusoidal_shape", shape_str(sin_out.data.get_shape()))

    # 12. PE(0,0) = sin(0) = 0.0 → output = 1.0 + 0.0 = 1.0
    emit("sinusoidal_pe_0_0", float_str(sin_out.data.get(0, 0)))

    # 13. PE(0,1) = cos(0) = 1.0 → output = 1.0 + 1.0 = 2.0
    emit("sinusoidal_pe_0_1", float_str(sin_out.data.get(0, 1)))

    # 14. PE(1,0) = sin(1) ≈ 0.841471 → output ≈ 1.841471
    emit("sinusoidal_pe_1_0", float_str(sin_out.data.get(1, 0)))

    # 15. No trainable parameters
    emit("sinusoidal_no_params", str(len(sin_pe.parameters())))

    # ============================================================
    # Part 4: Learned Positional Encoding
    # ============================================================

    learned_pe = PositionalEncoding("learned", 100, 8)

    # 16. Output shape
    learned_out = learned_pe.forward(Tensor.zeros(5, 8))
    emit("learned_shape", shape_str(learned_out.data.get_shape()))

    # 17. Has trainable parameters
    emit("learned_params_count", str(len(learned_pe.parameters())))

    # 18. Learned weight shape
    emit("learned_weight_shape", shape_str(learned_pe.parameters()[0].data.get_shape()))


if __name__ == "__main__":
    main()
