package dev.tensorhero.tinytorch;

import java.util.ArrayList;
import java.util.List;

/**
 * Layer — abstract base class for all neural network layers.
 *
 * <p>Provides common infrastructure: train/eval mode switching,
 * parameter collection, and child layer management. Subclasses
 * must implement {@link #forward(Tensor)}.</p>
 */
public abstract class Layer {

    /** Training mode flag. Default: true (training). Set to false for inference. */
    public boolean training = true;

    // ================================================================
    // E03 — Linear Layer
    // ================================================================

    /**
     * Forward pass through the layer.
     *
     * @param input the input tensor
     * @return the output tensor
     */
    public abstract Tensor forward(Tensor input);

    /**
     * Returns the list of trainable parameters in this layer.
     * Override in parametric layers (e.g. Linear).
     *
     * @return list of parameter tensors (default: empty)
     */
    public List<Tensor> parameters() {
        return new ArrayList<>();
    }

    /**
     * Returns the list of direct child layers.
     * Override in container layers (e.g. Sequential).
     *
     * @return list of child layers (default: empty)
     */
    public List<Layer> children() {
        return new ArrayList<>();
    }

    /**
     * Recursively sets this layer and all children to training mode.
     */
    public void train() {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Recursively sets this layer and all children to evaluation (inference) mode.
     */
    public void eval() {
        throw new UnsupportedOperationException("TODO: E03");
    }
}
