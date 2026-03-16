// TestE12.java — E12 Embeddings test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Embedding;
import dev.tensorhero.tinytorch.PositionalEncoding;
import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinynum.NDArray;

public class TestE12 {
    public static void main(String[] args) {
        // ============================================================
        // Part 1: Embedding — lookup & basic properties
        // ============================================================

        // Create embedding with deterministic weights for testing
        Embedding emb = new Embedding(4, 3);
        emb.weight = Tensor.fromArray(new float[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        }, 4, 3);
        emb.weight.requiresGrad = true;

        // 1. Shape of forward output
        Tensor out = emb.forward(new int[]{0, 2, 3});
        emit("embedding_shape", shapeStr(out.shape()));

        // 2. Lookup correctness — row 2 element [0,0]
        Tensor out2 = emb.forward(new int[]{2});
        emit("embedding_lookup_r0c0", floatStr(getAt2d(out2, 0, 0)));

        // 3. Lookup correctness — row 2 element [0,2]
        emit("embedding_lookup_r0c2", floatStr(getAt2d(out2, 0, 2)));

        // 4. Repeated index — both rows should be identical
        Tensor outRep = emb.forward(new int[]{1, 1});
        emit("embedding_repeated", floatStr(getAt2d(outRep, 0, 0)));

        // 5. Parameters count
        emit("embedding_params_count", String.valueOf(emb.parameters().size()));

        // 6. Weight shape
        emit("embedding_weight_shape", shapeStr(emb.weight.shape()));

        // ============================================================
        // Part 2: Embedding — backward & gradient accumulation
        // ============================================================

        // 7. Gradient shape after backward
        Embedding emb2 = new Embedding(4, 3);
        emb2.weight = Tensor.fromArray(new float[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        }, 4, 3);
        emb2.weight.requiresGrad = true;

        // forward([2]) → [[7,8,9]] shape [1,3]
        // sum over dim 1 → [24] shape [1]
        // sum over dim 0 → scalar
        Tensor fwd = emb2.forward(new int[]{2});
        Tensor loss = fwd.sum(1, false).sum(0, false);
        loss.backward();
        emit("embedding_grad_shape", shapeStr(emb2.weight.shape()));

        // 8. Gradient at looked-up row [2,0] = 1.0
        emit("embedding_grad_single", floatStr(emb2.weight.grad.data.get(2, 0)));

        // 9. Gradient at non-looked-up row [0,0] = 0.0
        emit("embedding_grad_zero", floatStr(emb2.weight.grad.data.get(0, 0)));

        // 10. Gradient accumulation with repeated index
        Embedding emb3 = new Embedding(4, 3);
        emb3.weight = Tensor.fromArray(new float[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        }, 4, 3);
        emb3.weight.requiresGrad = true;

        Tensor fwd3 = emb3.forward(new int[]{1, 1});
        Tensor loss3 = fwd3.sum(1, false).sum(0, false);
        loss3.backward();
        // Row 1 looked up twice → gradient accumulated: 2.0
        emit("embedding_grad_accumulate", floatStr(emb3.weight.grad.data.get(1, 0)));

        // ============================================================
        // Part 3: Sinusoidal Positional Encoding
        // ============================================================

        PositionalEncoding sinPE = new PositionalEncoding("sinusoidal", 100, 8);

        // 11. Output shape
        Tensor input = Tensor.ones(5, 8);
        Tensor sinOut = sinPE.forward(input);
        emit("sinusoidal_shape", shapeStr(sinOut.shape()));

        // 12. PE(0,0) = sin(0) = 0.0 → output = 1.0 + 0.0 = 1.0
        emit("sinusoidal_pe_0_0", floatStr(sinOut.data.get(0, 0)));

        // 13. PE(0,1) = cos(0) = 1.0 → output = 1.0 + 1.0 = 2.0
        emit("sinusoidal_pe_0_1", floatStr(sinOut.data.get(0, 1)));

        // 14. PE(1,0) = sin(1) ≈ 0.841471 → output = 1.0 + 0.841471 ≈ 1.841471
        emit("sinusoidal_pe_1_0", floatStr(sinOut.data.get(1, 0)));

        // 15. No trainable parameters
        emit("sinusoidal_no_params", String.valueOf(sinPE.parameters().size()));

        // ============================================================
        // Part 4: Learned Positional Encoding
        // ============================================================

        PositionalEncoding learnedPE = new PositionalEncoding("learned", 100, 8);

        // 16. Output shape
        Tensor learnedOut = learnedPE.forward(Tensor.zeros(5, 8));
        emit("learned_shape", shapeStr(learnedOut.shape()));

        // 17. Has trainable parameters
        emit("learned_params_count", String.valueOf(learnedPE.parameters().size()));

        // 18. Learned weight shape
        emit("learned_weight_shape",
            shapeStr(learnedPE.parameters().get(0).shape()));
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

    static float getAt2d(Tensor t, int row, int col) {
        return t.data.get(row, col);
    }
}
