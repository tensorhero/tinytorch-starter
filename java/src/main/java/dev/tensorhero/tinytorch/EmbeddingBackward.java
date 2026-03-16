package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Backward function for embedding lookup.
 *
 * <p>Accumulates gradients back to the weight matrix using scatter-add.
 * Repeated indices result in accumulated gradients.</p>
 */
public class EmbeddingBackward implements Function {

    private int[] indices;
    private int vocabSize;
    private int embedDim;
    private Tensor[] savedInputs;

    // ================================================================
    // E12 — Embeddings
    // ================================================================

    public EmbeddingBackward(int[] indices, int vocabSize, int embedDim) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    @Override
    public Tensor[] forward(Tensor... inputs) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    @Override
    public NDArray[] backward(NDArray gradOutput) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    @Override
    public Tensor[] inputs() {
        throw new UnsupportedOperationException("TODO: E12");
    }
}
