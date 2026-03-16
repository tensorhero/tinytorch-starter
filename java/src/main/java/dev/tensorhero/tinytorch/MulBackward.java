package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for element-wise multiplication: z = a * b.
 *
 * <p>Gradients: ∂z/∂a = b, ∂z/∂b = a</p>
 */
public class MulBackward implements Function {

    private Tensor[] savedInputs;

    @Override
    public Tensor[] forward(Tensor... inputs) {
        throw new UnsupportedOperationException("TODO: E05");
    }

    @Override
    public NDArray[] backward(NDArray gradOutput) {
        throw new UnsupportedOperationException("TODO: E05");
    }

    @Override
    public Tensor[] inputs() {
        throw new UnsupportedOperationException("TODO: E05");
    }
}
