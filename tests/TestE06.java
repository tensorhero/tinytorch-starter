// TestE06.java — E06 More Backward Ops test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify backward implementations
// for matMul, sum, mean, reshape, transpose, exp, log, activations, losses, and dropout.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Function;
import dev.tensorhero.tinytorch.Activations;
import dev.tensorhero.tinytorch.Losses;
import dev.tensorhero.tinytorch.Dropout;
import dev.tensorhero.tinynum.NDArray;

public class TestE06 {
    public static void main(String[] args) {

        NDArray gradOnes;

        // ===== Part 1: Tensor operations graph recording + backward =====

        // --- matMul ---
        Tensor a = Tensor.fromArray(new float[]{1,2,3,4}, 2, 2);
        a.requiresGrad = true;
        Tensor b = Tensor.fromArray(new float[]{5,6,7,8}, 2, 2);
        Tensor zMat = a.matMul(b);
        emit("graph_matmul_has_fn", String.valueOf(zMat.gradFn != null).toLowerCase());
        // forward: [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        emit("forward_matmul_00", floatStr(zMat.data.get(0, 0)));
        emit("forward_matmul_11", floatStr(zMat.data.get(1, 1)));
        // backward with ones(2,2): dA = ones @ B^T, dB = A^T @ ones
        gradOnes = NDArray.ones(2, 2);
        NDArray[] matGrads = zMat.gradFn.backward(gradOnes);
        // dA = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        emit("backward_matmul_da_00", floatStr(matGrads[0].get(0, 0)));
        emit("backward_matmul_da_01", floatStr(matGrads[0].get(0, 1)));
        // dB = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        emit("backward_matmul_db_00", floatStr(matGrads[1].get(0, 0)));
        emit("backward_matmul_db_10", floatStr(matGrads[1].get(1, 0)));

        // --- sum ---
        Tensor s = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        s.requiresGrad = true;
        Tensor zSum = s.sum(1, false);
        emit("graph_sum_has_fn", String.valueOf(zSum.gradFn != null).toLowerCase());
        // forward: sum axis=1 → [6, 15]
        emit("forward_sum_0", floatStr(zSum.data.get(0)));
        emit("forward_sum_1", floatStr(zSum.data.get(1)));
        // backward: broadcast [1,1] back → ones(2,3)
        NDArray[] sumGrads = zSum.gradFn.backward(NDArray.ones(2));
        emit("backward_sum_shape", shapeStr(sumGrads[0].shape()));
        emit("backward_sum_00", floatStr(sumGrads[0].get(0, 0)));

        // --- mean ---
        Tensor m = Tensor.fromArray(new float[]{2,4,6,8}, 2, 2);
        m.requiresGrad = true;
        Tensor zMean = m.mean(1, false);
        emit("graph_mean_has_fn", String.valueOf(zMean.gradFn != null).toLowerCase());
        // forward: mean axis=1 → [3, 7]
        emit("forward_mean_0", floatStr(zMean.data.get(0)));
        // backward: grad/2 broadcast → all 0.5
        NDArray[] meanGrads = zMean.gradFn.backward(NDArray.ones(2));
        emit("backward_mean_00", floatStr(meanGrads[0].get(0, 0)));

        // --- reshape ---
        Tensor r = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        r.requiresGrad = true;
        Tensor zReshape = r.reshape(3, 2);
        emit("graph_reshape_has_fn", String.valueOf(zReshape.gradFn != null).toLowerCase());
        emit("forward_reshape_shape", shapeStr(zReshape.data.shape()));
        // backward: reshape grad back to original shape
        NDArray[] reshapeGrads = zReshape.gradFn.backward(NDArray.ones(3, 2));
        emit("backward_reshape_shape", shapeStr(reshapeGrads[0].shape()));

        // --- transpose ---
        Tensor t = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        t.requiresGrad = true;
        Tensor zTrans = t.transpose(1, 0);
        emit("graph_transpose_has_fn", String.valueOf(zTrans.gradFn != null).toLowerCase());
        emit("forward_transpose_shape", shapeStr(zTrans.data.shape()));
        emit("forward_transpose_00", floatStr(zTrans.data.get(0, 0)));
        emit("forward_transpose_10", floatStr(zTrans.data.get(1, 0)));
        // backward: transpose grad with inverse perm
        NDArray[] transGrads = zTrans.gradFn.backward(NDArray.ones(3, 2));
        emit("backward_transpose_shape", shapeStr(transGrads[0].shape()));

        // --- exp ---
        Tensor e = Tensor.fromArray(new float[]{0.0f, 1.0f}, 2);
        e.requiresGrad = true;
        Tensor zExp = e.exp();
        emit("graph_exp_has_fn", String.valueOf(zExp.gradFn != null).toLowerCase());
        emit("forward_exp_0", floatStr(zExp.data.get(0)));
        emit("forward_exp_1", floatStr(zExp.data.get(1)));
        // backward: grad * exp(x) = [1*1, 1*e] = [1.0, 2.718...]
        NDArray[] expGrads = zExp.gradFn.backward(NDArray.ones(2));
        emit("backward_exp_0", floatStr(expGrads[0].get(0)));
        emit("backward_exp_1", floatStr(expGrads[0].get(1)));

        // --- log ---
        Tensor l = Tensor.fromArray(new float[]{1.0f, (float)Math.E}, 2);
        l.requiresGrad = true;
        Tensor zLog = l.log();
        emit("graph_log_has_fn", String.valueOf(zLog.gradFn != null).toLowerCase());
        emit("forward_log_0", floatStr(zLog.data.get(0)));
        emit("forward_log_1", floatStr(zLog.data.get(1)));
        // backward: grad / x = [1/1, 1/e]
        NDArray[] logGrads = zLog.gradFn.backward(NDArray.ones(2));
        emit("backward_log_0", floatStr(logGrads[0].get(0)));
        emit("backward_log_1", floatStr(logGrads[0].get(1)));

        // ===== Part 2: Activation backward =====

        Tensor ax = Tensor.fromArray(new float[]{-1.0f, 0.0f, 1.0f, 2.0f}, 4);
        ax.requiresGrad = true;

        // --- relu ---
        Tensor zRelu = Activations.relu(ax);
        emit("graph_relu_has_fn", String.valueOf(zRelu.gradFn != null).toLowerCase());
        emit("forward_relu_0", floatStr(zRelu.data.get(0)));
        emit("forward_relu_2", floatStr(zRelu.data.get(2)));
        NDArray[] reluGrads = zRelu.gradFn.backward(NDArray.ones(4));
        emit("backward_relu_0", floatStr(reluGrads[0].get(0)));   // x=-1 → 0
        emit("backward_relu_2", floatStr(reluGrads[0].get(2)));   // x=1 → 1

        // --- sigmoid ---
        Tensor zSig = Activations.sigmoid(ax);
        emit("graph_sigmoid_has_fn", String.valueOf(zSig.gradFn != null).toLowerCase());
        // sigmoid(0)=0.5
        emit("forward_sigmoid_1", floatStr(zSig.data.get(1)));
        NDArray[] sigGrads = zSig.gradFn.backward(NDArray.ones(4));
        // sigmoid'(0) = 0.5*(1-0.5) = 0.25
        emit("backward_sigmoid_1", floatStr(sigGrads[0].get(1)));

        // --- tanh ---
        Tensor zTanh = Activations.tanh(ax);
        emit("graph_tanh_has_fn", String.valueOf(zTanh.gradFn != null).toLowerCase());
        // tanh(0)=0
        emit("forward_tanh_1", floatStr(zTanh.data.get(1)));
        NDArray[] tanhGrads = zTanh.gradFn.backward(NDArray.ones(4));
        // tanh'(0) = 1 - 0^2 = 1
        emit("backward_tanh_1", floatStr(tanhGrads[0].get(1)));

        // --- gelu ---
        Tensor zGelu = Activations.gelu(ax);
        emit("graph_gelu_has_fn", String.valueOf(zGelu.gradFn != null).toLowerCase());
        // gelu(0) ≈ 0
        emit("forward_gelu_1", floatStr(zGelu.data.get(1)));
        NDArray[] geluGrads = zGelu.gradFn.backward(NDArray.ones(4));
        // gelu'(0) = 0.5
        emit("backward_gelu_1", floatStr(geluGrads[0].get(1)));

        // ===== Part 3: Loss backward =====

        // --- crossEntropy ---
        Tensor logits = Tensor.fromArray(new float[]{2.0f, 1.0f, 0.1f, 0.5f, 2.5f, 0.3f}, 2, 3);
        logits.requiresGrad = true;
        Tensor targets = Tensor.fromArray(new float[]{1,0,0, 0,1,0}, 2, 3);
        Tensor ce = Losses.crossEntropy(logits, targets);
        emit("graph_ce_has_fn", String.valueOf(ce.gradFn != null).toLowerCase());
        emit("forward_ce_shape", shapeStr(ce.data.shape()));
        NDArray[] ceGrads = ce.gradFn.backward(NDArray.ones(1));
        emit("backward_ce_shape", shapeStr(ceGrads[0].shape()));
        // Gradient should sum to ≈ 0 over classes for each sample
        float ceGradSum0 = ceGrads[0].get(0,0) + ceGrads[0].get(0,1) + ceGrads[0].get(0,2);
        emit("backward_ce_row_sum", floatStr(ceGradSum0));

        // --- mse (via Tensor ops) ---
        Tensor pred = Tensor.fromArray(new float[]{1.0f, 2.0f, 3.0f}, 3);
        pred.requiresGrad = true;
        Tensor target2 = Tensor.fromArray(new float[]{1.5f, 2.5f, 3.5f}, 3);
        Tensor mse = Losses.mse(pred, target2);
        emit("forward_mse_shape", shapeStr(mse.data.shape()));
        // MSE uses Tensor ops, so gradFn should exist on result
        emit("graph_mse_has_fn", String.valueOf(mse.gradFn != null).toLowerCase());

        // ===== Part 4: Dropout backward =====

        // Force a known mask for testing
        Tensor dx = Tensor.fromArray(new float[]{1,2,3,4}, 2, 2);
        dx.requiresGrad = true;
        NDArray mask = NDArray.fromArray(new float[]{2.0f, 0.0f, 2.0f, 0.0f}, 2, 2);
        dev.tensorhero.tinytorch.DropoutBackward dropFn = new dev.tensorhero.tinytorch.DropoutBackward(mask);
        Tensor dropOut = dropFn.forward(dx)[0];
        emit("forward_dropout_0", floatStr(dropOut.data.get(0, 0)));
        emit("forward_dropout_1", floatStr(dropOut.data.get(0, 1)));
        NDArray[] dropGrads = dropFn.backward(NDArray.ones(2, 2));
        emit("backward_dropout_0", floatStr(dropGrads[0].get(0, 0)));
        emit("backward_dropout_1", floatStr(dropGrads[0].get(0, 1)));
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }

    static String shapeStr(int[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        return sb.toString();
    }
}
