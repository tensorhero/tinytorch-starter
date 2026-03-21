package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for mean reduction: y = x.mean(axis, keepDims).
 *
 * <p>Gradient: broadcast grad / axisLength back to original shape.</p>
 */
public class MeanBackward implements Function {

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
