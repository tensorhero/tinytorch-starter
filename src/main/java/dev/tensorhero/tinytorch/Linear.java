package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Linear (fully connected) layer: y = x @ W.T + b.
 *
 * <p>The fundamental building block of neural networks. Applies a learned
 * linear transformation to incoming data.</p>
 *
 * <p>Weight shape: [outFeatures, inFeatures] (PyTorch convention).
 * Bias shape: [outFeatures]. Uses LeCun initialization.</p>
 */
public class Linear extends Layer {

    /** Weight matrix, shape [outFeatures, inFeatures]. */
    public Tensor weight;

    /** Bias vector, shape [outFeatures]. Null if bias=false. */
    public Tensor bias;

    /** Number of input features. */
    public int inFeatures;

    /** Number of output features. */
    public int outFeatures;

    // ================================================================
    // E03 — Linear Layer
    // ================================================================

    /**
     * Creates a Linear layer with bias.
     *
     * @param inFeatures  number of input features
     * @param outFeatures number of output features
     */
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    /**
     * Creates a Linear layer with optional bias.
     * Uses LeCun initialization: weight ~ N(0, 1/inFeatures), bias = zeros.
     *
     * @param inFeatures  number of input features
     * @param outFeatures number of output features
     * @param bias        whether to include a bias term
     */
    public Linear(int inFeatures, int outFeatures, boolean bias) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Forward pass: y = x @ W.T + b.
     *
     * @param input tensor of shape [batch, inFeatures]
     * @return tensor of shape [batch, outFeatures]
     */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("TODO: E03");
    }

    /**
     * Returns [weight] or [weight, bias] depending on configuration.
     */
    @Override
    public List<Tensor> parameters() {
        throw new UnsupportedOperationException("TODO: E03");
    }
}
