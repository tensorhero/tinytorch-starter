package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for element-wise addition: z = a + b.
 *
 * <p>Gradients: ∂z/∂a = 1, ∂z/∂b = 1</p>
 */
public class AddBackward implements Function {

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
