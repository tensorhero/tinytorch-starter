package dev.tensorhero.tinytorch;

import java.util.List;

/**
 * AdamW — Adam with decoupled weight decay.
 *
 * <p>Unlike standard L2 regularization, AdamW applies weight decay
 * independently before the gradient update. This is the standard
 * optimizer for GPT and modern LLM training.</p>
 */
public class AdamW extends Adam {

    // ================================================================
    // E08 — Optimizers
    // ================================================================

    /**
     * Creates an AdamW optimizer with default hyperparameters.
     *
     * @param params list of tensors to optimize
     * @param lr     learning rate
     */
    public AdamW(List<Tensor> params, float lr) {
        super(params, lr);
        // TODO: E08
    }

    /**
     * Creates an AdamW optimizer with custom hyperparameters.
     *
     * @param params      list of tensors to optimize
     * @param lr          learning rate
     * @param beta1       first moment decay rate
     * @param beta2       second moment decay rate
     * @param eps         numerical stability term
     * @param weightDecay weight decay coefficient
     */
    public AdamW(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay) {
        super(params, lr, beta1, beta2, eps);
        // TODO: E08
    }

    @Override
    public void step() {
        throw new UnsupportedOperationException("TODO: E08");
    }
}
