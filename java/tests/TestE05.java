// TestE05.java — E05 Computation Graph test driver
// Provided by tinytorch-starter. Do NOT modify.
// The tester compiles and runs this file to verify your backward classes
// and computation graph recording in Tensor operations.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Function;
import dev.tensorhero.tinynum.NDArray;

public class TestE05 {
    public static void main(String[] args) {

        // Shared inputs: x requires grad, y does not
        Tensor x = Tensor.fromArray(new float[]{2.0f, 3.0f}, 2);
        x.requiresGrad = true;
        Tensor y = Tensor.fromArray(new float[]{4.0f, 5.0f}, 2);

        // Gradient flowing back from downstream
        Tensor gradTensor = Tensor.ones(2);
        NDArray gradOut = gradTensor.data;

        // ===== Graph recording =====

        Tensor zAdd = x.add(y);
        emit("graph_add_has_fn", String.valueOf(zAdd.gradFn != null).toLowerCase());

        Tensor zSub = x.sub(y);
        emit("graph_sub_has_fn", String.valueOf(zSub.gradFn != null).toLowerCase());

        Tensor zMul = x.mul(y);
        emit("graph_mul_has_fn", String.valueOf(zMul.gradFn != null).toLowerCase());

        Tensor zDiv = x.div(y);
        emit("graph_div_has_fn", String.valueOf(zDiv.gradFn != null).toLowerCase());

        // No grad: both inputs have requiresGrad=false
        Tensor a = Tensor.fromArray(new float[]{1.0f, 2.0f}, 2);
        Tensor b = Tensor.fromArray(new float[]{3.0f, 4.0f}, 2);
        Tensor noGrad = a.add(b);
        emit("graph_no_grad", String.valueOf(noGrad.gradFn == null).toLowerCase());

        // requiresGrad propagation
        emit("graph_requires_grad", String.valueOf(zAdd.requiresGrad).toLowerCase());

        // inputs() returns saved inputs
        emit("graph_inputs_count", String.valueOf(zAdd.gradFn.inputs().length));

        // ===== Forward values (graph path should produce correct results) =====

        emit("forward_add_0", floatStr(zAdd.data.get(0)));
        emit("forward_add_1", floatStr(zAdd.data.get(1)));
        emit("forward_sub_0", floatStr(zSub.data.get(0)));
        emit("forward_mul_0", floatStr(zMul.data.get(0)));
        emit("forward_div_0", floatStr(zDiv.data.get(0)));

        // ===== Backward gradients =====

        // AddBackward: grad_a = gradOut, grad_b = gradOut
        NDArray[] addGrads = zAdd.gradFn.backward(gradOut);
        emit("backward_add_da", floatStr(addGrads[0].get(0)));
        emit("backward_add_db", floatStr(addGrads[1].get(0)));

        // SubBackward: grad_a = gradOut, grad_b = -gradOut
        NDArray[] subGrads = zSub.gradFn.backward(gradOut);
        emit("backward_sub_da", floatStr(subGrads[0].get(0)));
        emit("backward_sub_db", floatStr(subGrads[1].get(0)));

        // MulBackward: grad_a = gradOut * y, grad_b = gradOut * x
        NDArray[] mulGrads = zMul.gradFn.backward(gradOut);
        emit("backward_mul_da", floatStr(mulGrads[0].get(0)));
        emit("backward_mul_db", floatStr(mulGrads[1].get(0)));

        // DivBackward: grad_a = gradOut / y, grad_b = -gradOut * x / y²
        NDArray[] divGrads = zDiv.gradFn.backward(gradOut);
        emit("backward_div_da", floatStr(divGrads[0].get(0)));
        emit("backward_div_db", floatStr(divGrads[1].get(0)));
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }
}
