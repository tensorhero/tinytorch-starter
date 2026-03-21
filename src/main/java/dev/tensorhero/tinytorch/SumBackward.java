package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for sum reduction: y = x.sum(axis, keepDims).
 *
 * <p>Gradient: broadcast grad back to original shape.</p>
 */
public class SumBackward implements Function {

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
