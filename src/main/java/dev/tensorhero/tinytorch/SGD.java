package dev.tensorhero.tinytorch;

import java.util.List;

/**
 * SGD — Stochastic Gradient Descent with optional momentum.
 *
 * <p>Without momentum: {@code param -= lr * grad}.</p>
 * <p>With momentum: {@code v = momentum * v + grad; param -= lr * v}.</p>
 */
public class SGD extends Optimizer {

    // ================================================================
    // E08 — Optimizers
    // ================================================================

    /**
     * Creates an SGD optimizer without momentum.
     *
     * @param params list of tensors to optimize
     * @param lr     learning rate
     */
    public SGD(List<Tensor> params, float lr) {
        super(params);
        // TODO: E08
    }

    /**
     * Creates an SGD optimizer with momentum.
     *
     * @param params   list of tensors to optimize
     * @param lr       learning rate
     * @param momentum momentum factor (e.g. 0.9)
     */
    public SGD(List<Tensor> params, float lr, float momentum) {
        super(params);
        // TODO: E08
    }

    @Override
    public void step() {
        throw new UnsupportedOperationException("TODO: E08");
    }
}
