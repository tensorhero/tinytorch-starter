package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Embedding — maps token IDs to dense vectors via table lookup.
 *
 * <p>Maintains a weight matrix of shape [vocabSize, embedDim].
 * Forward pass selects rows by index. Backward pass accumulates
 * gradients via scatter-add.</p>
 */
public class Embedding {

    /** Weight matrix, shape [vocabSize, embedDim]. */
    public Tensor weight;

    /** Vocabulary size. */
    public int vocabSize;

    /** Embedding dimension. */
    public int embedDim;

    // ================================================================
    // E12 — Embeddings
    // ================================================================

    /**
     * Creates an Embedding layer with Xavier initialization.
     *
     * @param vocabSize number of tokens in vocabulary
     * @param embedDim  dimension of each embedding vector
     */
    public Embedding(int vocabSize, int embedDim) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    /**
     * Looks up embedding vectors for the given token indices.
     *
     * @param indices array of token IDs
     * @return tensor of shape [indices.length, embedDim]
     */
    public Tensor forward(int[] indices) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    /**
     * Returns trainable parameters (the weight matrix).
     */
    public List<Tensor> parameters() {
        throw new UnsupportedOperationException("TODO: E12");
    }
}
