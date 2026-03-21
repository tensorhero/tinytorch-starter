package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for transpose: y = x.transpose(axes).
 *
 * <p>Gradient: grad.transpose(inversePermutation(axes)).</p>
 */
public class TransposeBackward implements Function {

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
