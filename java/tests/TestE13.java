// TestE13.java — E13 Attention test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Attention;
import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinynum.NDArray;

public class TestE13 {
    public static void main(String[] args) {
        // ============================================================
        // Part 1: Causal Mask
        // ============================================================

        NDArray mask = Attention.createCausalMask(4);

        // 1. Shape
        emit("causal_mask_shape", shapeStr(mask.shape()));

        // 2. Diagonal — token sees itself (not masked)
        emit("causal_mask_self_visible", floatStr(mask.get(0, 0)));

        // 3. Upper triangle — future blocked
        emit("causal_mask_future_blocked", floatStr(mask.get(0, 1)));

        // 4. Lower triangle — past visible
        emit("causal_mask_past_visible", floatStr(mask.get(2, 1)));

        // 5. Last element — last token sees itself
        emit("causal_mask_last_self", floatStr(mask.get(3, 3)));

        // ============================================================
        // Part 2: SDPA — weight row sum verification
        // ============================================================

        // If V = ones, output[i,j] = sum(weights[i,:]) = 1.0 (softmax sums to 1)
        Tensor qOnes = Tensor.ones(3, 2);
        Tensor kOnes = Tensor.ones(3, 2);
        Tensor vOnes = Tensor.ones(3, 2);

        // 6. Without mask — weight sum
        Tensor wsOut = Attention.scaledDotProductAttention(qOnes, kOnes, vOnes, null);
        emit("sdpa_weight_sum_shape", shapeStr(wsOut.shape()));

        // 7. All output elements should be 1.0 (weight rows sum to 1)
        emit("sdpa_weight_sum_value", floatStr(wsOut.data.get(1, 0)));

        // 8. With causal mask — weight rows still sum to 1
        NDArray mask3 = Attention.createCausalMask(3);
        Tensor wsCausal = Attention.scaledDotProductAttention(qOnes, kOnes, vOnes, mask3);
        emit("sdpa_causal_weight_sum", floatStr(wsCausal.data.get(0, 0)));

        // ============================================================
        // Part 3: SDPA — correctness
        // ============================================================

        // Q = K = ones(3,4) → uniform attention; V = [1..12] shape 3,4
        Tensor qUnif = Tensor.ones(3, 4);
        Tensor kUnif = Tensor.ones(3, 4);
        Tensor vSeq = Tensor.fromArray(new float[]{
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        }, 3, 4);

        // 9. No mask — uniform average: output[0,0] = (1+5+9)/3 = 5.0
        Tensor noMaskOut = Attention.scaledDotProductAttention(qUnif, kUnif, vSeq, null);
        emit("sdpa_no_mask_value_00", floatStr(noMaskOut.data.get(0, 0)));

        // 10. No mask — output[0,1] = (2+6+10)/3 = 6.0
        emit("sdpa_no_mask_value_01", floatStr(noMaskOut.data.get(0, 1)));

        // With causal mask:
        // Row 0: only sees V[0] → [1,2,3,4]
        // Row 1: sees V[0],V[1] avg → [3,4,5,6]
        // Row 2: sees all → same as no mask → [5,6,7,8]
        NDArray mask3b = Attention.createCausalMask(3);
        Tensor causalOut = Attention.scaledDotProductAttention(qUnif, kUnif, vSeq, mask3b);

        // 11. Causal: row 0 only sees itself → V[0,0] = 1.0
        emit("sdpa_causal_row0_0", floatStr(causalOut.data.get(0, 0)));

        // 12. Causal: row 0 col 1 → V[0,1] = 2.0
        emit("sdpa_causal_row0_1", floatStr(causalOut.data.get(0, 1)));

        // 13. Causal: row 1 col 0 → (V[0,0]+V[1,0])/2 = (1+5)/2 = 3.0
        emit("sdpa_causal_row1_0", floatStr(causalOut.data.get(1, 0)));

        // 14. Causal: last row same as no mask → 5.0
        emit("sdpa_causal_last_same", floatStr(causalOut.data.get(2, 0)));

        // ============================================================
        // Part 4: Multi-Head Attention
        // ============================================================

        Attention.MultiHeadAttention mha = new Attention.MultiHeadAttention(8, 2);

        // 15. Output shape: [B=1, T=3, D=8]
        Tensor mhaInput = Tensor.ones(1, 3, 8);
        Tensor mhaOut = mha.forward(mhaInput);
        emit("mha_output_shape", shapeStr(mhaOut.shape()));

        // 16. Parameters count: 4 Linear × 2 params each = 8
        emit("mha_params_count", String.valueOf(mha.parameters().size()));

        // 17. Children count: 4 linear layers
        emit("mha_children_count", String.valueOf(mha.children().size()));
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String shapeStr(int[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        return sb.toString();
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }
}
