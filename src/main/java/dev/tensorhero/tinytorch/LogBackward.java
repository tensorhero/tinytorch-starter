package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for log: y = log(x).
 *
 * <p>Gradient: grad / x.</p>
 */
public class LogBackward implements Function {

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
