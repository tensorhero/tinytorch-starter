// TestE02.java — E02 Activations test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify your Activations implementation.

import dev.tensorhero.tinytorch.Activations;
import dev.tensorhero.tinytorch.Tensor;

public class TestE02 {
    public static void main(String[] args) {

        // ===== ReLU =====

        // relu basic: [-1, 0, 1, 2] → [0, 0, 1, 2]
        Tensor rx = Tensor.fromArray(new float[]{-1, 0, 1, 2}, 4);
        Tensor ry = Activations.relu(rx);
        emit("relu_basic", ry.toString());

        // relu all negative: [-3, -2, -1] → [0, 0, 0]
        Tensor rn = Tensor.fromArray(new float[]{-3, -2, -1}, 3);
        emit("relu_all_negative", Activations.relu(rn).toString());

        // relu preserves shape
        Tensor r2d = Tensor.fromArray(new float[]{-1, 2, -3, 4}, 2, 2);
        emit("relu_shape", shapeStr(Activations.relu(r2d).shape()));

        // ===== Sigmoid =====

        // sigmoid(0) = 0.5
        Tensor sz = Tensor.fromArray(new float[]{0}, 1);
        emit("sigmoid_zero", floatStr(getFirst(Activations.sigmoid(sz))));

        // sigmoid basic: [-2, 0, 2]
        Tensor sx = Tensor.fromArray(new float[]{-2, 0, 2}, 3);
        Tensor sy = Activations.sigmoid(sx);
        emit("sigmoid_neg2", floatStr(getAt(sy, 0)));
        emit("sigmoid_pos2", floatStr(getAt(sy, 2)));

        // sigmoid symmetry: sigmoid(-x) + sigmoid(x) ≈ 1.0
        Tensor s5 = Tensor.fromArray(new float[]{5}, 1);
        Tensor sn5 = Tensor.fromArray(new float[]{-5}, 1);
        float sigSum = getFirst(Activations.sigmoid(s5)) + getFirst(Activations.sigmoid(sn5));
        emit("sigmoid_symmetry_sum", floatStr(sigSum));

        // ===== Tanh =====

        // tanh(0) = 0.0
        Tensor tz = Tensor.fromArray(new float[]{0}, 1);
        emit("tanh_zero", floatStr(getFirst(Activations.tanh(tz))));

        // tanh basic: [0, 1, -1]
        Tensor tx = Tensor.fromArray(new float[]{0, 1, -1}, 3);
        Tensor ty = Activations.tanh(tx);
        emit("tanh_pos1", floatStr(getAt(ty, 1)));
        emit("tanh_neg1", floatStr(getAt(ty, 2)));

        // tanh antisymmetry: tanh(-x) = -tanh(x)
        Tensor t3 = Tensor.fromArray(new float[]{3}, 1);
        Tensor tn3 = Tensor.fromArray(new float[]{-3}, 1);
        float tanhSum = getFirst(Activations.tanh(t3)) + getFirst(Activations.tanh(tn3));
        emit("tanh_antisymmetry_sum", floatStr(tanhSum));

        // ===== GELU =====

        // gelu(0) = 0.0
        Tensor gz = Tensor.fromArray(new float[]{0}, 1);
        emit("gelu_zero", floatStr(getFirst(Activations.gelu(gz))));

        // gelu basic: [-1, 0, 1]
        Tensor gx = Tensor.fromArray(new float[]{-1, 0, 1}, 3);
        Tensor gy = Activations.gelu(gx);
        emit("gelu_neg1", floatStr(getAt(gy, 0)));
        emit("gelu_pos1", floatStr(getAt(gy, 2)));

        // gelu preserves large positive values approximately
        Tensor g3 = Tensor.fromArray(new float[]{3}, 1);
        emit("gelu_pos3", floatStr(getFirst(Activations.gelu(g3))));

        // ===== Softmax =====

        // softmax basic: [1, 2, 3] along axis=0
        Tensor smx = Tensor.fromArray(new float[]{1, 2, 3}, 3);
        Tensor smy = Activations.softmax(smx, 0);
        emit("softmax_val0", floatStr(getAt(smy, 0)));
        emit("softmax_val1", floatStr(getAt(smy, 1)));
        emit("softmax_val2", floatStr(getAt(smy, 2)));

        // softmax sum ≈ 1.0
        float smSum = getAt(smy, 0) + getAt(smy, 1) + getAt(smy, 2);
        emit("softmax_sum", floatStr(smSum));

        // softmax uniform: [1, 1, 1] → [1/3, 1/3, 1/3]
        Tensor smu = Tensor.fromArray(new float[]{1, 1, 1}, 3);
        Tensor smuy = Activations.softmax(smu, 0);
        emit("softmax_uniform_val", floatStr(getAt(smuy, 0)));

        // softmax numerical stability: [1000, 1001, 1002] should not overflow
        Tensor sml = Tensor.fromArray(new float[]{1000, 1001, 1002}, 3);
        Tensor smly = Activations.softmax(sml, 0);
        float smlSum = getAt(smly, 0) + getAt(smly, 1) + getAt(smly, 2);
        emit("softmax_stability_sum", floatStr(smlSum));
        emit("softmax_stability_val2", floatStr(getAt(smly, 2)));

        // softmax 2D: [[1,2,3],[1,1,1]] along axis=1
        Tensor sm2d = Tensor.fromArray(new float[]{1,2,3,1,1,1}, 2, 3);
        Tensor sm2dy = Activations.softmax(sm2d, 1);
        emit("softmax_2d_shape", shapeStr(sm2dy.shape()));

        // row 0 sum ≈ 1.0
        float row0sum = getAt2d(sm2dy, 0, 0) + getAt2d(sm2dy, 0, 1) + getAt2d(sm2dy, 0, 2);
        emit("softmax_2d_row0_sum", floatStr(row0sum));

        // row 1 should be uniform (all ≈ 1/3)
        emit("softmax_2d_row1_val", floatStr(getAt2d(sm2dy, 1, 0)));
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

    /** Get the first element of a 1D tensor. */
    static float getFirst(Tensor t) {
        return t.data.get(0);
    }

    /** Get element at index i of a 1D tensor. */
    static float getAt(Tensor t, int i) {
        return t.data.get(i);
    }

    /** Get element at (row, col) of a 2D tensor. */
    static float getAt2d(Tensor t, int row, int col) {
        return t.data.get(row, col);
    }
}
