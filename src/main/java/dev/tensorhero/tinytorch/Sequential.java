package dev.tensorhero.tinytorch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Sequential container — chains multiple layers into a single model.
 *
 * <p>Input is passed through each layer in order. Parameters and
 * train/eval mode are managed recursively across all child layers.</p>
 */
public class Sequential extends Layer {

    /** The ordered list of child layers. */
    public Layer[] layers;

    // ================================================================
    // E03 — Linear Layer
    // ================================================================

    /**
     * Creates a Sequential container with the given layers.
     *
     * @param layers the layers to chain, in order
     */
    public Sequential(Layer... layers) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Forward pass: passes input through each layer sequentially.
     *
     * @param input the input tensor
     * @return the output of the last layer
     */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Collects parameters from all child layers.
     */
    @Override
    public List<Tensor> parameters() {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Returns all child layers.
     */
    @Override
    public List<Layer> children() {
        throw new UnsupportedOperationException("TODO: E03");
    }
}
