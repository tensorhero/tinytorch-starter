package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Autograd function — records forward computation and computes backward gradients.
 *
 * <p>This interface is a placeholder for E01-E04. Full implementation starts in E05.</p>
 */
public interface Function {

    // ================================================================
    // E05 — Computation Graph (placeholder)
    // ================================================================

    /**
     * Forward pass: compute output tensors from inputs.
     *
     * @param inputs the input tensors
     * @return the output tensors
     */
    Tensor[] forward(Tensor... inputs);

    /**
     * Backward pass: compute gradients of inputs given gradient of output.
     *
     * @param gradOutput the gradient of the loss w.r.t. the output
     * @return gradients of the loss w.r.t. each input (as raw NDArrays)
     */
    NDArray[] backward(NDArray gradOutput);

    /**
     * Returns the input tensors saved during forward (for backward computation).
     *
     * @return the saved input tensors
     */
    Tensor[] inputs();
}
