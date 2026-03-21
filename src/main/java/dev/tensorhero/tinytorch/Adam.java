package dev.tensorhero.tinytorch;

import java.util.List;

/**
 * Adam — Adaptive Moment Estimation optimizer.
 *
 * <p>Maintains first moment (mean) and second moment (variance) estimates
 * with bias correction. Default hyperparameters: lr=0.001, β₁=0.9,
 * β₂=0.999, ε=1e-8.</p>
 */
public class Adam extends Optimizer {

    /** Learning rate. */
    protected float lr;

    /** First moment decay rate. */
    protected float beta1;

    /** Second moment decay rate. */
    protected float beta2;

    /** Numerical stability term. */
    protected float eps;

    /** Step counter for bias correction. */
    protected int t;

    // ================================================================
    // E08 — Optimizers
    // ================================================================

    /**
     * Creates an Adam optimizer with default hyperparameters.
     *
     * @param params list of tensors to optimize
     * @param lr     learning rate
     */
    public Adam(List<Tensor> params, float lr) {
        super(params);
        // TODO: E08
    }

    /**
     * Creates an Adam optimizer with custom hyperparameters.
     *
     * @param params list of tensors to optimize
     * @param lr     learning rate
     * @param beta1  first moment decay rate
     * @param beta2  second moment decay rate
     * @param eps    numerical stability term
     */
    public Adam(List<Tensor> params, float lr, float beta1, float beta2, float eps) {
        super(params);
        // TODO: E08
    }

    @Override
    public void step() {
        throw new UnsupportedOperationException("TODO: E08");
    }
}
