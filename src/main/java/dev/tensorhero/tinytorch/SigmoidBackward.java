package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for Sigmoid: y = 1 / (1 + exp(-x)).
 *
 * <p>Gradient: grad * y * (1 - y). Saves output for efficient backward.</p>
 */
public class SigmoidBackward implements Function {

    @Override
    public Tensor[] forward(Tensor... inputs) {
        throw new UnsupportedOperationException("TODO: E06");
    }

    @Override
    public NDArray[] backward(NDArray gradOutput) {
        throw new UnsupportedOperationException("TODO: E06");
    }

    @Override
    public Tensor[] inputs() {
        throw new UnsupportedOperationException("TODO: E06");
    }
}
