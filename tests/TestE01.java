// TestE01.java — E01 Tensor Class test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify your Tensor implementation.

import dev.tensorhero.tinytorch.Tensor;

public class TestE01 {
    public static void main(String[] args) {
        // --- zeros ---
        Tensor z = Tensor.zeros(2, 3);
        emit("zeros_size", String.valueOf(z.size()));
        emit("zeros_ndim", String.valueOf(z.ndim()));
        emit("zeros_shape", shapeStr(z.shape()));
        emit("zeros_toString", z.toString());

        // --- ones ---
        Tensor o = Tensor.ones(3, 4);
        emit("ones_size", String.valueOf(o.size()));
        emit("ones_shape", shapeStr(o.shape()));
        emit("ones_toString", Tensor.ones(2, 3).toString());

        // --- fromArray 1D ---
        Tensor a1d = Tensor.fromArray(new float[]{1,2,3}, 3);
        emit("fromArray_1d_toString", a1d.toString());

        // --- fromArray 2D ---
        Tensor a = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        emit("fromArray_2d_toString", a.toString());

        // --- full ---
        Tensor f = Tensor.full(7.0f, 2, 2);
        emit("full_toString", f.toString());

        // --- add ---
        Tensor x = Tensor.fromArray(new float[]{1,2,3,4}, 2, 2);
        Tensor y = Tensor.ones(2, 2);
        emit("add_toString", x.add(y).toString());

        // --- sub ---
        emit("sub_toString", x.sub(y).toString());

        // --- mul ---
        emit("mul_toString", x.mul(y).toString());

        // --- div ---
        Tensor d = Tensor.fromArray(new float[]{2,4,6,8}, 2, 2);
        Tensor two = Tensor.full(2.0f, 2, 2);
        emit("div_toString", d.div(two).toString());

        // --- matMul ---
        Tensor m1 = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        Tensor m2 = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 3, 2);
        Tensor prod = m1.matMul(m2);
        emit("matMul_shape", shapeStr(prod.shape()));
        emit("matMul_toString", prod.toString());

        // --- sum ---
        Tensor s = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        emit("sum_axis0", s.sum(0, false).toString());
        emit("sum_axis1", s.sum(1, false).toString());

        // --- mean ---
        emit("mean_axis0_keepDims", s.mean(0, true).toString());
        emit("mean_axis1", s.mean(1, false).toString());

        // --- reshape ---
        Tensor r = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        emit("reshape_shape", shapeStr(r.reshape(3, 2).shape()));
        emit("reshape_toString", r.reshape(3, 2).toString());

        // --- transpose ---
        Tensor t = Tensor.fromArray(new float[]{1,2,3,4,5,6}, 2, 3);
        emit("transpose_shape", shapeStr(t.transpose(1, 0).shape()));
        emit("transpose_toString", t.transpose(1, 0).toString());

        // --- gradient fields (dormant) ---
        Tensor g = Tensor.zeros(3, 3);
        emit("requiresGrad", String.valueOf(g.requiresGrad));
        emit("grad_null", String.valueOf(g.grad == null));
        emit("gradFn_null", String.valueOf(g.gradFn == null));

        // --- randn shape ---
        Tensor rn = Tensor.randn(4, 5);
        emit("randn_shape", shapeStr(rn.shape()));
        emit("randn_size", String.valueOf(rn.size()));
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
}
