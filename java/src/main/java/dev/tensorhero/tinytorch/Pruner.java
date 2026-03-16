package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;
import java.util.List;

/**
 * Weight pruning utilities for model compression.
 */
public class Pruner {

    /**
     * Magnitude pruning: zero out the smallest weights by absolute value.
     *
     * @param weight   the weight tensor to prune (modified in-place)
     * @param sparsity fraction of weights to prune, in [0.0, 1.0]
     */
    public static void magnitudePrune(Tensor weight, float sparsity) {
        throw new UnsupportedOperationException("TODO: E17");
    }

    /**
     * Measure the sparsity of a model (percentage of zero-valued parameters).
     * Only counts parameters with ndim >= 2 (skips biases).
     *
     * @param model the model to measure
     * @return sparsity percentage in [0.0, 100.0]
     */
    public static float measureSparsity(Layer model) {
        throw new UnsupportedOperationException("TODO: E17");
    }
}
