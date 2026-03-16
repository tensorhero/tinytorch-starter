"""test_e13.py — E13 Attention test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.attention import (
    create_causal_mask,
    scaled_dot_product_attention,
    MultiHeadAttention,
)


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    # ============================================================
    # Part 1: Causal Mask
    # ============================================================

    mask = create_causal_mask(4)

    # 1. Shape
    emit("causal_mask_shape", shape_str(mask.get_shape()))

    # 2. Diagonal — token sees itself (not masked)
    emit("causal_mask_self_visible", float_str(mask.get(0, 0)))

    # 3. Upper triangle — future blocked
    emit("causal_mask_future_blocked", float_str(mask.get(0, 1)))

    # 4. Lower triangle — past visible
    emit("causal_mask_past_visible", float_str(mask.get(2, 1)))

    # 5. Last element — last token sees itself
    emit("causal_mask_last_self", float_str(mask.get(3, 3)))

    # ============================================================
    # Part 2: SDPA — weight row sum verification
    # ============================================================

    # If V = ones, output[i,j] = sum(weights[i,:]) = 1.0 (softmax sums to 1)
    q_ones = Tensor.ones(3, 2)
    k_ones = Tensor.ones(3, 2)
    v_ones = Tensor.ones(3, 2)

    # 6. Without mask — weight sum
    ws_out = scaled_dot_product_attention(q_ones, k_ones, v_ones, None)
    emit("sdpa_weight_sum_shape", shape_str(ws_out.data.get_shape()))

    # 7. All output elements should be 1.0 (weight rows sum to 1)
    emit("sdpa_weight_sum_value", float_str(ws_out.data.get(1, 0)))

    # 8. With causal mask — weight rows still sum to 1
    mask3 = create_causal_mask(3)
    ws_causal = scaled_dot_product_attention(q_ones, k_ones, v_ones, mask3)
    emit("sdpa_causal_weight_sum", float_str(ws_causal.data.get(0, 0)))

    # ============================================================
    # Part 3: SDPA — correctness
    # ============================================================

    # Q = K = ones(3,4) -> uniform attention; V = [1..12] shape 3,4
    q_unif = Tensor.ones(3, 4)
    k_unif = Tensor.ones(3, 4)
    v_seq = Tensor.from_array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4
    )

    # 9. No mask — uniform average: output[0,0] = (1+5+9)/3 = 5.0
    no_mask_out = scaled_dot_product_attention(q_unif, k_unif, v_seq, None)
    emit("sdpa_no_mask_value_00", float_str(no_mask_out.data.get(0, 0)))

    # 10. No mask — output[0,1] = (2+6+10)/3 = 6.0
    emit("sdpa_no_mask_value_01", float_str(no_mask_out.data.get(0, 1)))

    # With causal mask:
    # Row 0: only sees V[0] -> [1,2,3,4]
    # Row 1: sees V[0],V[1] avg -> [3,4,5,6]
    # Row 2: sees all -> same as no mask -> [5,6,7,8]
    mask3b = create_causal_mask(3)
    causal_out = scaled_dot_product_attention(q_unif, k_unif, v_seq, mask3b)

    # 11. Causal: row 0 only sees itself -> V[0,0] = 1.0
    emit("sdpa_causal_row0_0", float_str(causal_out.data.get(0, 0)))

    # 12. Causal: row 0 col 1 -> V[0,1] = 2.0
    emit("sdpa_causal_row0_1", float_str(causal_out.data.get(0, 1)))

    # 13. Causal: row 1 col 0 -> (V[0,0]+V[1,0])/2 = (1+5)/2 = 3.0
    emit("sdpa_causal_row1_0", float_str(causal_out.data.get(1, 0)))

    # 14. Causal: last row same as no mask -> 5.0
    emit("sdpa_causal_last_same", float_str(causal_out.data.get(2, 0)))

    # ============================================================
    # Part 4: Multi-Head Attention
    # ============================================================

    mha = MultiHeadAttention(8, 2)

    # 15. Output shape: [B=1, T=3, D=8]
    mha_input = Tensor.ones(1, 3, 8)
    mha_out = mha.forward(mha_input)
    emit("mha_output_shape", shape_str(mha_out.data.get_shape()))

    # 16. Parameters count: 4 Linear × 2 params each = 8
    emit("mha_params_count", str(len(mha.parameters())))

    # 17. Children count: 4 linear layers
    emit("mha_children_count", str(len(mha.children())))


if __name__ == "__main__":
    main()
