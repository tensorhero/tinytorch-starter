// TestE03.java — E03 Linear Layer test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify your Layer implementations.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Layer;
import dev.tensorhero.tinytorch.Linear;
import dev.tensorhero.tinytorch.Dropout;
import dev.tensorhero.tinytorch.Sequential;

public class TestE03 {
    public static void main(String[] args) {

        // ===== Linear basics =====

        Linear lin = new Linear(3, 2);
        emit("linear_weight_shape", shapeStr(lin.weight.shape()));
        emit("linear_bias_shape", shapeStr(lin.bias.shape()));
        emit("linear_bias_init", lin.bias.toString());
        emit("linear_params_count", String.valueOf(lin.parameters().size()));

        Linear linNb = new Linear(3, 2, false);
        emit("linear_no_bias_params", String.valueOf(linNb.parameters().size()));
        emit("linear_no_bias_null", String.valueOf(linNb.bias == null));

        // ===== Linear forward with known weights =====

        // W = [[1,0,0],[0,1,0]]  shape [2,3]
        // b = [10, 20]
        lin.weight = Tensor.fromArray(new float[]{1,0,0, 0,1,0}, 2, 3);
        lin.bias = Tensor.fromArray(new float[]{10, 20}, 2);

        // x = [[1,2,3],[4,5,6]]  shape [2,3]
        // y = x @ W.T + b = [[1,2],[4,5]] + [10,20] = [[11,22],[14,25]]
        Tensor x = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        Tensor y = lin.forward(x);
        emit("linear_forward_shape", shapeStr(y.shape()));
        emit("linear_forward_toString", y.toString());

        // Without bias
        linNb.weight = Tensor.fromArray(new float[]{1,0,0, 0,1,0}, 2, 3);
        Tensor yNb = linNb.forward(Tensor.fromArray(new float[]{1,2,3}, 1, 3));
        emit("linear_forward_no_bias", yNb.toString());

        // ===== LeCun init variance check =====

        Linear linBig = new Linear(1000, 50);
        Tensor w = linBig.weight;
        // Compute global mean (keepdim=true to avoid 0D scalar)
        float wMean = w.mean(1, true).mean(0, true).data.get(0, 0);
        // Compute global variance
        Tensor diff = w.sub(Tensor.full(wMean, w.shape()));
        Tensor diffSq = diff.mul(diff);
        float wVar = diffSq.mean(1, true).mean(0, true).data.get(0, 0);
        emit("linear_init_variance", floatStr(wVar));

        // ===== Dropout =====

        Tensor dx = Tensor.fromArray(new float[]{1,2,3,4}, 2, 2);

        // Eval mode: identity
        Dropout dropEval = new Dropout(0.5f);
        dropEval.eval();
        emit("dropout_eval", dropEval.forward(dx).toString());

        // p=0 in train mode: identity
        Dropout drop0 = new Dropout(0.0f);
        emit("dropout_p0_train", drop0.forward(dx).toString());

        // p=1 in train mode: all zeros
        Dropout drop1 = new Dropout(1.0f);
        emit("dropout_p1_train", drop1.forward(dx).toString());

        // Shape preserved (using eval mode for determinism)
        emit("dropout_shape", shapeStr(dropEval.forward(dx).shape()));

        // No parameters
        emit("dropout_no_params", String.valueOf(new Dropout(0.5f).parameters().size()));

        // ===== Sequential =====

        Linear sl1 = new Linear(3, 2);
        sl1.weight = Tensor.fromArray(new float[]{1,0,0, 0,1,0}, 2, 3);
        sl1.bias = Tensor.fromArray(new float[]{0, 0}, 2);

        Linear sl2 = new Linear(2, 1);
        sl2.weight = Tensor.fromArray(new float[]{1, 1}, 1, 2);
        sl2.bias = Tensor.fromArray(new float[]{0}, 1);

        Sequential seq = new Sequential(sl1, sl2);

        // forward: [[1,2,3]] → sl1 → [[1,2]] → sl2 → [[3]]
        Tensor sx = Tensor.fromArray(new float[]{1,2,3}, 1, 3);
        Tensor sy = seq.forward(sx);
        emit("sequential_forward_shape", shapeStr(sy.shape()));
        emit("sequential_forward", sy.toString());

        // Parameters: sl1(weight+bias) + sl2(weight+bias) = 4
        emit("sequential_params_count", String.valueOf(seq.parameters().size()));

        // Children
        emit("sequential_children_count", String.valueOf(seq.children().size()));

        // ===== Train / Eval mode =====

        emit("training_default", String.valueOf(new Linear(3, 2).training));

        Linear linMode = new Linear(3, 2);
        linMode.eval();
        emit("eval_sets_false", String.valueOf(linMode.training));
        linMode.train();
        emit("train_sets_true", String.valueOf(linMode.training));

        // Recursive mode switching
        seq.eval();
        emit("eval_recursive", String.valueOf(seq.children().get(0).training));
        seq.train();
        emit("train_recursive", String.valueOf(seq.children().get(0).training));
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
