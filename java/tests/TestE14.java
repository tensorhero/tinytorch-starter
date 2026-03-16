// TestE14.java — E14 Transformer Block test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.TransformerBlock;
import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinynum.NDArray;

public class TestE14 {
    public static void main(String[] args) {
        // ============================================================
        // Part 1: LayerNorm
        // ============================================================

        TransformerBlock.LayerNorm ln = new TransformerBlock.LayerNorm(4);

        // x = [[1,2,3,4]], shape [1,4]
        Tensor lnInput = Tensor.fromArray(new float[]{1, 2, 3, 4}, 1, 4);
        Tensor lnOut = ln.forward(lnInput);

        // 1. Output shape preserved
        emit("ln_output_shape", shapeStr(lnOut.shape()));

        // 2. Specific output value: normalized[0,0] = (1-2.5)/sqrt(1.25+eps) ≈ -1.3416
        emit("ln_output_value_00", floatStr(lnOut.data.get(0, 0)));

        // 3. Specific output value: normalized[0,3] = (4-2.5)/sqrt(1.25+eps) ≈ 1.3416
        emit("ln_output_value_03", floatStr(lnOut.data.get(0, 3)));

        // 4. Mean of output row ≈ 0 (beta=0)
        float rowMean = (lnOut.data.get(0, 0) + lnOut.data.get(0, 1)
                + lnOut.data.get(0, 2) + lnOut.data.get(0, 3)) / 4.0f;
        emit("ln_output_mean", floatStr(rowMean));

        // 5. Parameters count
        emit("ln_params_count", String.valueOf(ln.parameters().size()));

        // ============================================================
        // Part 1b: LayerNorm Backward
        // ============================================================

        TransformerBlock.LayerNorm lnBwd = new TransformerBlock.LayerNorm(4);
        Tensor lnBwdInput = Tensor.fromArray(new float[]{1, 2, 3, 4}, 1, 4);
        lnBwdInput.requiresGrad = true;
        Tensor lnBwdOut = lnBwd.forward(lnBwdInput);
        // loss = sum of all output elements
        Tensor lnLoss = lnBwdOut.sum(1, false).sum(0, false);
        lnLoss.backward();

        // 6. beta.grad[0] = 1.0 (one sample, upstream grad = 1)
        emit("ln_backward_beta_grad_0", floatStr(lnBwd.beta.grad.data.get(0)));

        // 7. gamma.grad[0] ≈ -1.3416 (= normalized[0,0])
        emit("ln_backward_gamma_grad_0", floatStr(lnBwd.gamma.grad.data.get(0)));

        // ============================================================
        // Part 1c: LayerNorm — uniform input (edge case)
        // ============================================================

        TransformerBlock.LayerNorm lnUnif = new TransformerBlock.LayerNorm(4);
        Tensor unifInput = Tensor.fromArray(new float[]{5, 5, 5, 5}, 1, 4);
        Tensor unifOut = lnUnif.forward(unifInput);

        // 8. Uniform input: diff=0, normalized=0 → output = 0*gamma+beta = beta = 0
        emit("ln_uniform_output_00", floatStr(unifOut.data.get(0, 0)));

        // ============================================================
        // Part 2: MLP
        // ============================================================

        TransformerBlock.MLP mlp = new TransformerBlock.MLP(8);
        Tensor mlpInput = Tensor.ones(1, 3, 8);
        Tensor mlpOut = mlp.forward(mlpInput);

        // 9. Output shape preserved
        emit("mlp_output_shape", shapeStr(mlpOut.shape()));

        // 10. Parameters count: fc1(weight+bias) + fc2(weight+bias) = 4
        emit("mlp_params_count", String.valueOf(mlp.parameters().size()));

        // 11. Children count: fc1, fc2
        emit("mlp_children_count", String.valueOf(mlp.children().size()));

        // ============================================================
        // Part 3: TransformerBlock
        // ============================================================

        TransformerBlock.Block block = new TransformerBlock.Block(8, 2);
        Tensor blockInput = Tensor.ones(1, 3, 8);
        Tensor blockOut = block.forward(blockInput);

        // 12. Output shape preserved (residual connection)
        emit("block_output_shape", shapeStr(blockOut.shape()));

        // 13. Parameters count: ln1(2) + attn(8) + ln2(2) + mlp(4) = 16
        emit("block_params_count", String.valueOf(block.parameters().size()));

        // 14. Children count: ln1, attn, ln2, mlp = 4
        emit("block_children_count", String.valueOf(block.children().size()));

        // ============================================================
        // Part 3b: Residual connection verification
        // ============================================================

        // 15. With all-zeros input: residual adds back, output should be non-trivial
        //     (if residual works, zeros + attn(ln(zeros)) + mlp(ln(...)) ≠ zeros
        //     because ln(zeros)=beta*gamma and biases are non-zero after linear init)
        Tensor zeroInput = Tensor.zeros(1, 3, 8);
        Tensor zeroOut = block.forward(zeroInput);
        // If residual didn't work, output would be just mlp output, not input+mlp
        emit("block_residual_shape", shapeStr(zeroOut.shape()));
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
