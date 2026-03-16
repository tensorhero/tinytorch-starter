package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Losses — loss functions for training neural networks.
 *
 * <p>All methods are static and return a scalar {@link Tensor} of shape [1].
 * Forward-only in E04; backward implementations will be added in E06.</p>
 */
public class Losses {

    // ================================================================
    // E04 — Loss Functions
    // ================================================================

    /**
     * Numerically stable log-softmax along the given axis.
     *
     * <p>logSoftmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))</p>
     */
    public static Tensor logSoftmax(Tensor x, int axis) {
        throw new UnsupportedOperationException("TODO: E04");
    }

    /**
     * Mean Squared Error: mean((pred - target)^2).
     *
     * @return scalar Tensor of shape [1]
     */
    public static Tensor mse(Tensor pred, Tensor target) {
        throw new UnsupportedOperationException("TODO: E04");
    }

    /**
     * Cross-entropy loss with one-hot encoded targets.
     *
     * <p>CE = -mean(sum(targets * logSoftmax(logits), axis=classes))</p>
     *
     * @param logits  raw scores, shape [batch, numClasses]
     * @param targets one-hot encoded, shape [batch, numClasses]
     * @return scalar Tensor of shape [1]
     */
    public static Tensor crossEntropy(Tensor logits, Tensor targets) {
        throw new UnsupportedOperationException("TODO: E04");
    }

    // ================================================================
    // E06 — More Backward Ops
    // ================================================================
    // crossEntropy: when logits.requiresGrad, create CrossEntropyBackward,
    //   call fn.forward(), set result.requiresGrad = true, result.gradFn = fn.
    //   Uses fused gradient: dL/dlogits = (softmax(logits) - targets) / batchSize.
    // mse: rewrite to use Tensor ops (sub → mul → reshape → mean) so the
    //   computation graph is recorded automatically. No MSEBackward needed.
}
