// TestE08.java — E08 Optimizers test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.SGD;
import dev.tensorhero.tinytorch.Adam;
import dev.tensorhero.tinytorch.AdamW;

import java.util.Arrays;
import java.util.List;

public class TestE08 {
    public static void main(String[] args) {
        // =============================================================
        // Test 1–2: SGD basic step
        // param = [2.0, 4.0], grad = [1.0, 2.0], lr = 0.5
        // After step: param = [1.5, 3.0]
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{2.0f, 4.0f}, 2);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{1.0f, 2.0f}, 2);

            SGD sgd = new SGD(Arrays.asList(p), 0.5f);
            sgd.step();

            emit("sgd_step_param0", floatStr(getAt(p, 0)));
            emit("sgd_step_param1", floatStr(getAt(p, 1)));
        }

        // =============================================================
        // Test 3: SGD zeroGrad
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{1.0f}, 1);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{0.5f}, 1);

            SGD sgd = new SGD(Arrays.asList(p), 0.1f);
            sgd.zeroGrad();

            emit("sgd_zero_grad", String.valueOf(p.grad == null));
        }

        // =============================================================
        // Test 4–5: SGD with momentum (two steps)
        // param = [1.0], lr = 0.1, momentum = 0.9
        // Step 1: grad=[2.0], v = 2.0, param = 1.0 - 0.1*2.0 = 0.8
        // Step 2: grad=[2.0], v = 0.9*2.0 + 2.0 = 3.8, param = 0.8 - 0.1*3.8 = 0.42
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{1.0f}, 1);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{2.0f}, 1);

            SGD sgd = new SGD(Arrays.asList(p), 0.1f, 0.9f);
            sgd.step();
            emit("sgd_momentum_step1", floatStr(getAt(p, 0)));

            // Reset grad for step 2
            p.grad = Tensor.fromArray(new float[]{2.0f}, 1);
            sgd.step();
            emit("sgd_momentum_step2", floatStr(getAt(p, 0)));
        }

        // =============================================================
        // Test 6: SGD skips params without gradient
        // =============================================================
        {
            Tensor p1 = Tensor.fromArray(new float[]{5.0f}, 1);
            p1.requiresGrad = true;
            // p1.grad is null — should be skipped

            Tensor p2 = Tensor.fromArray(new float[]{3.0f}, 1);
            p2.requiresGrad = true;
            p2.grad = Tensor.fromArray(new float[]{1.0f}, 1);

            SGD sgd = new SGD(Arrays.asList(p1, p2), 0.5f);
            sgd.step();

            // p1 unchanged, p2 updated
            emit("sgd_no_grad_skip", String.valueOf(getAt(p1, 0) == 5.0f));
        }

        // =============================================================
        // Test 7–8: Adam basic (two steps)
        // param = [3.0], grad = [1.0], lr = 0.01
        // Step 1: m_hat=1.0, v_hat=1.0, param ≈ 3.0 - 0.01 = 2.99
        // Step 2: m_hat=1.0, v_hat=1.0, param ≈ 2.99 - 0.01 = 2.98
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{3.0f}, 1);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{1.0f}, 1);

            Adam adam = new Adam(Arrays.asList(p), 0.01f);
            adam.step();
            emit("adam_step1", floatStr(getAt(p, 0)));

            p.grad = Tensor.fromArray(new float[]{1.0f}, 1);
            adam.step();
            emit("adam_step2", floatStr(getAt(p, 0)));
        }

        // =============================================================
        // Test 9: AdamW weight decay
        // param = [2.0], grad = [1.0], lr = 0.01, weightDecay = 0.5
        // Weight decay: param = 2.0 * (1 - 0.01*0.5) = 1.99
        // Adam update: param = 1.99 - 0.01 ≈ 1.98
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{2.0f}, 1);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{1.0f}, 1);

            AdamW adamw = new AdamW(Arrays.asList(p), 0.01f, 0.9f, 0.999f, 1e-8f, 0.5f);
            adamw.step();
            emit("adamw_step1", floatStr(getAt(p, 0)));
        }

        // =============================================================
        // Test 10: AdamW result < Adam result (weight decay effect)
        // =============================================================
        {
            // Adam
            Tensor pAdam = Tensor.fromArray(new float[]{2.0f}, 1);
            pAdam.requiresGrad = true;
            pAdam.grad = Tensor.fromArray(new float[]{1.0f}, 1);
            Adam adam = new Adam(Arrays.asList(pAdam), 0.01f);
            adam.step();

            // AdamW
            Tensor pAdamW = Tensor.fromArray(new float[]{2.0f}, 1);
            pAdamW.requiresGrad = true;
            pAdamW.grad = Tensor.fromArray(new float[]{1.0f}, 1);
            AdamW adamw = new AdamW(Arrays.asList(pAdamW), 0.01f, 0.9f, 0.999f, 1e-8f, 0.5f);
            adamw.step();

            // AdamW should produce smaller param due to weight decay
            emit("adamw_smaller_than_adam", String.valueOf(getAt(pAdamW, 0) < getAt(pAdam, 0)));
        }

        // =============================================================
        // Test 11–13: Multiple parameters
        // p1 = [1.0, 2.0], grad = [0.5, 0.5]
        // p2 = [3.0], grad = [1.0]
        // SGD lr = 0.2
        // After step: p1 = [0.9, 1.9], p2 = [2.8]
        // =============================================================
        {
            Tensor p1 = Tensor.fromArray(new float[]{1.0f, 2.0f}, 2);
            p1.requiresGrad = true;
            p1.grad = Tensor.fromArray(new float[]{0.5f, 0.5f}, 2);

            Tensor p2 = Tensor.fromArray(new float[]{3.0f}, 1);
            p2.requiresGrad = true;
            p2.grad = Tensor.fromArray(new float[]{1.0f}, 1);

            SGD sgd = new SGD(Arrays.asList(p1, p2), 0.2f);
            sgd.step();

            emit("multi_params_p1_0", floatStr(getAt(p1, 0)));
            emit("multi_params_p1_1", floatStr(getAt(p1, 1)));
            emit("multi_params_p2_0", floatStr(getAt(p2, 0)));
        }
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }

    static float getAt(Tensor t, int i) {
        return t.data.get(i);
    }
}
