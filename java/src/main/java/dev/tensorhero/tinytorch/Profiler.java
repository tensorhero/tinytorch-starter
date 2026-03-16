package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;
import java.util.List;

/**
 * Performance profiler for counting parameters and estimating FLOPs.
 */
public class Profiler {

    /**
     * Count the total number of trainable parameters in a model.
     * Recursively traverses all children layers.
     */
    public static int countParams(Layer model) {
        throw new UnsupportedOperationException("TODO: E17");
    }

    /**
     * Estimate the total FLOPs for a forward pass.
     * Only counts Linear layer FLOPs: 2 × inFeatures × outFeatures.
     * Recursively traverses all children layers.
     *
     * @param model      the model to profile
     * @param inputShape the input tensor shape, e.g. [batchSize, seqLen, dim]
     * @return estimated FLOPs
     */
    public static long countFlops(Layer model, int[] inputShape) {
        throw new UnsupportedOperationException("TODO: E17");
    }
}
