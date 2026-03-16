package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for element-wise division: z = a / b.
 *
 * <p>Gradients: ∂z/∂a = 1/b, ∂z/∂b = -a/b²</p>
 */
public class DivBackward implements Function {

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
