package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for cross-entropy loss (fused gradient).
 *
 * <p>Uses the fused formula: dL/dlogits = (softmax(logits) - targets) / batchSize.
 * No need for a separate SoftmaxBackward.</p>
 */
public class CrossEntropyBackward implements Function {

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
