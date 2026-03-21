// TestE09.java — E09 Training Loop test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Trainer;
import dev.tensorhero.tinytorch.Layer;
import dev.tensorhero.tinytorch.Losses;
import dev.tensorhero.tinytorch.SGD;

import java.util.Arrays;
import java.util.List;

public class TestE09 {
    public static void main(String[] args) {
        // =============================================================
        // Test 1–4: Cosine schedule
        // =============================================================
        {
            float lr0 = Trainer.cosineSchedule(0, 100, 0.1f, 0.001f);
            emit("cosine_at_start", floatStr(lr0));

            float lr50 = Trainer.cosineSchedule(50, 100, 0.1f, 0.001f);
            emit("cosine_at_middle", floatStr(lr50));

            float lr100 = Trainer.cosineSchedule(100, 100, 0.1f, 0.001f);
            emit("cosine_at_end", floatStr(lr100));

            float lr25 = Trainer.cosineSchedule(25, 100, 0.1f, 0.001f);
            emit("cosine_at_quarter", floatStr(lr25));
        }

        // =============================================================
        // Test 5–7: clipGradNorm — clipping applied
        // grad = [3.0, 4.0] → norm = 5.0, maxNorm = 2.5
        // After clip: grad = [1.5, 2.0], returns 5.0
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{0.0f, 0.0f}, 2);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{3.0f, 4.0f}, 2);

            float norm = Trainer.clipGradNorm(Arrays.asList(p), 2.5f);
            emit("clip_returns_norm", floatStr(norm));
            emit("clip_after_grad0", floatStr(getAt(p.grad, 0)));
            emit("clip_after_grad1", floatStr(getAt(p.grad, 1)));
        }

        // =============================================================
        // Test 8: clipGradNorm — no clipping needed
        // grad = [1.0, 0.0] → norm = 1.0, maxNorm = 5.0
        // =============================================================
        {
            Tensor p = Tensor.fromArray(new float[]{0.0f, 0.0f}, 2);
            p.requiresGrad = true;
            p.grad = Tensor.fromArray(new float[]{1.0f, 0.0f}, 2);

            Trainer.clipGradNorm(Arrays.asList(p), 5.0f);
            emit("clip_no_clip_grad0", floatStr(getAt(p.grad, 0)));
        }

        // =============================================================
        // Test 9: Accuracy — all correct
        // pred argmax = [0, 1], target argmax = [0, 1] → 1.0
        // =============================================================
        {
            Tensor pred = Tensor.fromArray(new float[]{0.9f, 0.1f, 0.2f, 0.8f}, 2, 2);
            Tensor target = Tensor.fromArray(new float[]{1.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
            float acc = Trainer.accuracy(pred, target);
            emit("accuracy_perfect", floatStr(acc));
        }

        // =============================================================
        // Test 10: Accuracy — half correct
        // pred argmax = [0, 0], target argmax = [0, 1] → 0.5
        // =============================================================
        {
            Tensor pred = Tensor.fromArray(new float[]{0.9f, 0.1f, 0.8f, 0.2f}, 2, 2);
            Tensor target = Tensor.fromArray(new float[]{1.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
            float acc = Trainer.accuracy(pred, target);
            emit("accuracy_partial", floatStr(acc));
        }

        // =============================================================
        // Test 11–12: Training step
        // Simple model: pred = input + w, MSE loss, SGD lr=0.1
        // w starts at 2.0, input = [1.0], target = [5.0]
        // loss1 = (1+2-5)^2 = 4.0 > 0
        // After 10 steps, loss decreases
        // =============================================================
        {
            Layer model = new Layer() {
                Tensor w;
                {
                    w = Tensor.fromArray(new float[]{2.0f}, 1);
                    w.requiresGrad = true;
                }
                @Override
                public Tensor forward(Tensor input) {
                    return input.add(w);
                }
                @Override
                public List<Tensor> parameters() {
                    return Arrays.asList(w);
                }
            };

            SGD optimizer = new SGD(model.parameters(), 0.1f);
            Trainer trainer = new Trainer(model, optimizer, Losses::mse);

            Tensor x = Tensor.fromArray(new float[]{1.0f}, 1);
            Tensor y = Tensor.fromArray(new float[]{5.0f}, 1);

            float loss1 = trainer.trainStep(x, y);
            emit("train_step_positive", String.valueOf(loss1 > 0));

            // Run 9 more steps
            for (int i = 0; i < 9; i++) {
                trainer.trainStep(x, y);
            }
            float loss10 = trainer.trainStep(x, y);
            emit("train_step_decreasing", String.valueOf(loss10 < loss1));
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

    static float getAt2d(Tensor t, int row, int col) {
        return t.data.get(row, col);
    }
}
