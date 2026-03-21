package dev.tensorhero.tinytorch;

import java.util.ArrayList;
import java.util.List;

/**
 * Optimizer — abstract base class for all parameter optimizers.
 *
 * <p>Stores a list of trainable parameters and provides {@link #zeroGrad()}
 * to clear gradients. Subclasses implement {@link #step()} to update
 * parameters using their gradients.</p>
 */
public abstract class Optimizer {

    /** The list of trainable parameters managed by this optimizer. */
    protected List<Tensor> params;

    // ================================================================
    // E08 — Optimizers
    // ================================================================

    /**
     * Creates an optimizer for the given parameters.
     *
     * @param params list of tensors to optimize
     */
    public Optimizer(List<Tensor> params) {
        // TODO: E08
    }

    /**
     * Clears the gradients of all parameters (sets grad to null).
     */
    public void zeroGrad() {
        throw new UnsupportedOperationException("TODO: E08");
    }

    /**
     * Performs a single optimization step (parameter update).
     */
    public abstract void step();
}
