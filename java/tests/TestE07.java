// TestE07.java — E07 Backpropagation test driver
// Provided by tinytorch-starter. Do NOT modify.
// Tests: backward(), topologicalSort(), reduceBroadcastGrad(), noGrad().

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Activations;
import dev.tensorhero.tinytorch.Losses;
import dev.tensorhero.tinynum.NDArray;
import java.util.List;

public class TestE07 {
    public static void main(String[] args) {

        // ===== Part 1: Topological Sort =====

        Tensor x = Tensor.fromArray(new float[]{1, 2, 3, 4}, 2, 2);
        x.requiresGrad = true;
        Tensor w = Tensor.fromArray(new float[]{0.5f, -0.5f, 0.5f, -0.5f}, 2, 2);
        w.requiresGrad = true;
        Tensor y = x.matMul(w);  // y = x @ w
        List<Tensor> order = Tensor.topologicalSort(y);
        // order should be [x, w, y] or [w, x, y] — root is last
        emit("topo_size", String.valueOf(order.size()));
        emit("topo_root_last", String.valueOf(order.get(order.size() - 1) == y).toLowerCase());
        // x and w should come before y
        int xIdx = order.indexOf(x);
        int wIdx = order.indexOf(w);
        int yIdx = order.indexOf(y);
        emit("topo_x_before_y", String.valueOf(xIdx < yIdx).toLowerCase());
        emit("topo_w_before_y", String.valueOf(wIdx < yIdx).toLowerCase());

        // ===== Part 2: Simple backward (y = x * w, scalar) =====

        Tensor a = Tensor.fromArray(new float[]{3.0f}, 1);
        a.requiresGrad = true;
        Tensor b = Tensor.fromArray(new float[]{4.0f}, 1);
        b.requiresGrad = true;
        Tensor c = a.mul(b);  // c = a * b = 12
        c.backward();
        // dc/da = b = 4, dc/db = a = 3
        emit("simple_backward_a_grad", floatStr(a.grad.data.get(0)));
        emit("simple_backward_b_grad", floatStr(b.grad.data.get(0)));

        // ===== Part 3: Multi-op backward (y = (x @ W + b).relu().sum().mean()) =====

        Tensor x2 = Tensor.fromArray(new float[]{1, 2, 3, 4}, 2, 2);
        x2.requiresGrad = true;
        Tensor w2 = Tensor.fromArray(new float[]{0.5f, -0.5f, 0.5f, -0.5f}, 2, 2);
        w2.requiresGrad = true;
        Tensor b2 = Tensor.fromArray(new float[]{0.1f, -0.1f}, 2);
        b2.requiresGrad = true;

        Tensor mm = x2.matMul(w2);                     // [2,2]
        Tensor added = mm.add(b2);                      // [2,2] + [2] broadcast
        Tensor activated = Activations.relu(added);     // [2,2]
        Tensor summed = activated.sum(1, false);         // [2]
        Tensor loss = summed.mean(0, true);             // scalar [1]

        loss.backward();

        // Check that all grads exist
        emit("multi_x_grad_exists", String.valueOf(x2.grad != null).toLowerCase());
        emit("multi_w_grad_exists", String.valueOf(w2.grad != null).toLowerCase());
        emit("multi_b_grad_exists", String.valueOf(b2.grad != null).toLowerCase());

        // Check grad shapes
        emit("multi_x_grad_shape", shapeStr(x2.grad.data.shape()));
        emit("multi_w_grad_shape", shapeStr(w2.grad.data.shape()));
        emit("multi_b_grad_shape", shapeStr(b2.grad.data.shape()));

        // Verify numerically: compute forward values first
        // mm = [[1*0.5+2*0.5, 1*(-0.5)+2*(-0.5)], [3*0.5+4*0.5, 3*(-0.5)+4*(-0.5)]]
        //    = [[1.5, -1.5], [3.5, -3.5]]
        // added = [[1.6, -1.6], [3.6, -3.6]]
        // relu  = [[1.6, 0], [3.6, 0]]
        // sum(1) = [1.6, 3.6]
        // mean(0) = 2.6
        emit("multi_loss_value", floatStr(loss.data.get(0)));

        // b2.grad shape should be [2] (broadcast reduced from [2,2])
        // b2.grad[0] should reflect that only positive activations contribute
        emit("multi_b_grad_0", floatStr(b2.grad.data.get(0)));
        emit("multi_b_grad_1", floatStr(b2.grad.data.get(1)));

        // ===== Part 4: Broadcast gradient reduction =====

        Tensor x3 = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x3.requiresGrad = true;
        Tensor b3 = Tensor.fromArray(new float[]{0.1f, 0.2f, 0.3f}, 3);
        b3.requiresGrad = true;
        Tensor sum3 = x3.add(b3).sum(1, false).mean(0, true);
        sum3.backward();

        emit("broadcast_b_grad_shape", shapeStr(b3.grad.data.shape()));
        // Each element of b3 is added to a column, then summed along axis 1 and meaned.
        // d(sum)/d(b3[j]) = 1 for each of 2 rows, mean divides by 2 → b3.grad = [1, 1, 1]
        emit("broadcast_b_grad_0", floatStr(b3.grad.data.get(0)));
        emit("broadcast_b_grad_1", floatStr(b3.grad.data.get(1)));
        emit("broadcast_b_grad_2", floatStr(b3.grad.data.get(2)));

        // ===== Part 5: Gradient accumulation (x used twice: z = x + x) =====

        Tensor x4 = Tensor.fromArray(new float[]{1.0f, 2.0f, 3.0f}, 3);
        x4.requiresGrad = true;
        Tensor z4 = x4.add(x4);  // z = x + x = 2x
        Tensor s4 = z4.sum(0, true);
        s4.backward();
        // ds/dx = 2 for each element
        emit("accum_grad_0", floatStr(x4.grad.data.get(0)));
        emit("accum_grad_1", floatStr(x4.grad.data.get(1)));
        emit("accum_grad_2", floatStr(x4.grad.data.get(2)));

        // ===== Part 6: noGrad =====

        Tensor x5 = Tensor.fromArray(new float[]{1.0f, 2.0f}, 2);
        x5.requiresGrad = true;
        Tensor y5 = Tensor.fromArray(new float[]{3.0f, 4.0f}, 2);

        Tensor result = Tensor.noGrad(() -> x5.add(y5));
        emit("nograd_result_0", floatStr(result.data.get(0)));
        emit("nograd_result_1", floatStr(result.data.get(1)));
        emit("nograd_no_fn", String.valueOf(result.gradFn == null).toLowerCase());

        // Verify gradEnabled is restored after noGrad
        Tensor z5 = x5.add(y5);  // should have gradFn again
        emit("nograd_restored", String.valueOf(z5.gradFn != null).toLowerCase());

        // ===== Part 7: Finite difference verification =====
        // Verify backward produces correct gradient by comparing with numerical gradient.
        // f(x) = sum(x^2) where x = [1, 2, 3]
        // Analytical: df/dx = 2x = [2, 4, 6]

        Tensor xFd = Tensor.fromArray(new float[]{1.0f, 2.0f, 3.0f}, 3);
        xFd.requiresGrad = true;
        Tensor sqr = xFd.mul(xFd);          // x^2
        Tensor sFd = sqr.sum(0, true);     // sum(x^2) = 14
        sFd.backward();
        emit("fd_grad_0", floatStr(xFd.grad.data.get(0)));  // 2*1 = 2
        emit("fd_grad_1", floatStr(xFd.grad.data.get(1)));  // 2*2 = 4
        emit("fd_grad_2", floatStr(xFd.grad.data.get(2)));  // 2*3 = 6
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
