package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Dropout layer for regularization.
 *
 * <p>During training: randomly zeros elements with probability p,
 * scales survivors by 1/(1-p) (inverted dropout).
 * During inference: passes input through unchanged.</p>
 *
 * <p>The dropout mask is saved as {@code lastMask} for use in
 * backward pass (E06).</p>
 */
public class Dropout extends Layer {

    /** Dropout probability (fraction of elements to zero). */
    public float p;

    /** The mask applied during the last forward pass (for backward in E06). */
    public NDArray lastMask;

    // ================================================================
    // E03 — Linear Layer
    // ================================================================

    /**
     * Creates a Dropout layer.
     *
     * @param p probability of zeroing each element (0.0 to 1.0)
     */
    public Dropout(float p) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Forward pass: applies inverted dropout in training mode,
     * identity in eval mode.
     *
     * @param input the input tensor
     * @return the output tensor (same shape as input)
     */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    // ================================================================
    // E06 — More Backward Ops
    // ================================================================
    // When input.requiresGrad and training, create DropoutBackward,
    // call fn.forward(), and set result.requiresGrad = true, result.gradFn = fn.
    // DropoutBackward reuses lastMask saved during forward.
}
