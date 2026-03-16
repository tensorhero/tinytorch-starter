// TestE04.java — E04 Loss Functions test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify your Losses implementations.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Losses;

public class TestE04 {
    public static void main(String[] args) {

        // ===== MSE =====

        // Perfect prediction → loss = 0
        Tensor p1 = Tensor.fromArray(new float[]{1, 2, 3}, 3);
        Tensor t1 = Tensor.fromArray(new float[]{1, 2, 3}, 3);
        Tensor mse1 = Losses.mse(p1, t1);
        emit("mse_zero_loss", floatStr(mse1.data.get(0)));

        // Basic: pred=[1,0], target=[0,1] → MSE=1.0
        Tensor p2 = Tensor.fromArray(new float[]{1, 0}, 2);
        Tensor t2 = Tensor.fromArray(new float[]{0, 1}, 2);
        Tensor mse2 = Losses.mse(p2, t2);
        emit("mse_basic", floatStr(mse2.data.get(0)));

        // Known: diff=[-0.5,0.5,0.0,1.0] → MSE=0.375
        Tensor p3 = Tensor.fromArray(new float[]{2.5f, 0.0f, 2.0f, 8.0f}, 4);
        Tensor t3 = Tensor.fromArray(new float[]{3.0f, -0.5f, 2.0f, 7.0f}, 4);
        Tensor mse3 = Losses.mse(p3, t3);
        emit("mse_known", floatStr(mse3.data.get(0)));

        // 2D: same data reshaped to [2,2] → same result
        Tensor p4 = Tensor.fromArray(new float[]{2.5f, 0.0f, 2.0f, 8.0f}, 2, 2);
        Tensor t4 = Tensor.fromArray(new float[]{3.0f, -0.5f, 2.0f, 7.0f}, 2, 2);
        Tensor mse4 = Losses.mse(p4, t4);
        emit("mse_2d", floatStr(mse4.data.get(0)));

        // Shape: should be [1]
        emit("mse_shape", shapeStr(mse3.shape()));

        // ===== logSoftmax =====

        // logSoftmax([1,2,3], axis=0) → first element ≈ -2.4076
        Tensor lx1 = Tensor.fromArray(new float[]{1, 2, 3}, 3);
        Tensor ls1 = Losses.logSoftmax(lx1, 0);
        emit("logsoftmax_val0", floatStr(ls1.data.get(0)));

        // sum(exp(logSoftmax)) ≈ 1.0
        float sumExp = 0;
        for (int i = 0; i < 3; i++) {
            sumExp += (float) Math.exp(ls1.data.get(i));
        }
        emit("logsoftmax_sum_exp", floatStr(sumExp));

        // Uniform: logSoftmax([1,1,1], axis=0)[0] = log(1/3) ≈ -1.0986
        Tensor lx2 = Tensor.fromArray(new float[]{1, 1, 1}, 3);
        Tensor ls2 = Losses.logSoftmax(lx2, 0);
        emit("logsoftmax_uniform", floatStr(ls2.data.get(0)));

        // Numerical stability: large values don't overflow
        Tensor lx3 = Tensor.fromArray(new float[]{1000, 1001, 1002}, 3);
        Tensor ls3 = Losses.logSoftmax(lx3, 0);
        float sumExpStable = 0;
        for (int i = 0; i < 3; i++) {
            sumExpStable += (float) Math.exp(ls3.data.get(i));
        }
        emit("logsoftmax_stability", floatStr(sumExpStable));

        // 2D shape preserved
        Tensor lx4 = Tensor.fromArray(new float[]{1,2,3, 4,5,6}, 2, 3);
        Tensor ls4 = Losses.logSoftmax(lx4, 1);
        emit("logsoftmax_shape", shapeStr(ls4.shape()));

        // All logSoftmax values ≤ 0
        boolean allNeg = true;
        for (int i = 0; i < 3; i++) {
            if (ls1.data.get(i) > 1e-6) {
                allNeg = false;
                break;
            }
        }
        emit("logsoftmax_negative", String.valueOf(allNeg));

        // ===== CrossEntropy =====

        // Perfect prediction → loss ≈ 0
        Tensor ceLogits1 = Tensor.fromArray(new float[]{100, 0, 0}, 1, 3);
        Tensor ceTargets1 = Tensor.fromArray(new float[]{1, 0, 0}, 1, 3);
        Tensor ce1 = Losses.crossEntropy(ceLogits1, ceTargets1);
        emit("ce_perfect", floatStr(ce1.data.get(0)));

        // Uniform logits → loss = log(3) ≈ 1.0986
        Tensor ceLogits2 = Tensor.fromArray(new float[]{1, 1, 1}, 1, 3);
        Tensor ceTargets2 = Tensor.fromArray(new float[]{1, 0, 0}, 1, 3);
        Tensor ce2 = Losses.crossEntropy(ceLogits2, ceTargets2);
        emit("ce_uniform", floatStr(ce2.data.get(0)));

        // Known: logits=[2,1,0.1], targets=[1,0,0] → CE ≈ 0.4170
        Tensor ceLogits3 = Tensor.fromArray(new float[]{2, 1, 0.1f}, 1, 3);
        Tensor ceTargets3 = Tensor.fromArray(new float[]{1, 0, 0}, 1, 3);
        Tensor ce3 = Losses.crossEntropy(ceLogits3, ceTargets3);
        emit("ce_known", floatStr(ce3.data.get(0)));

        // Batch of 2: average over samples
        Tensor ceLogits4 = Tensor.fromArray(new float[]{2,1,0.1f, 0.5f,2.5f,0.3f}, 2, 3);
        Tensor ceTargets4 = Tensor.fromArray(new float[]{1,0,0, 0,1,0}, 2, 3);
        Tensor ce4 = Losses.crossEntropy(ceLogits4, ceTargets4);
        emit("ce_batch", floatStr(ce4.data.get(0)));

        // Wrong prediction → high loss
        Tensor ceLogits5 = Tensor.fromArray(new float[]{0, 0, 2}, 1, 3);
        Tensor ceTargets5 = Tensor.fromArray(new float[]{1, 0, 0}, 1, 3);
        Tensor ce5 = Losses.crossEntropy(ceLogits5, ceTargets5);
        emit("ce_wrong", floatStr(ce5.data.get(0)));

        // Shape: should be [1]
        emit("ce_shape", shapeStr(ce3.shape()));
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
