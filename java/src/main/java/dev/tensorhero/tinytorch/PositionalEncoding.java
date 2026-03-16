package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * PositionalEncoding — adds position information to token embeddings.
 *
 * <p>Supports two modes:</p>
 * <ul>
 *   <li>"sinusoidal" — fixed sin/cos table (no trainable parameters)</li>
 *   <li>"learned" — trainable position embedding (uses Embedding internally)</li>
 * </ul>
 */
public class PositionalEncoding {

    /** Encoding mode: "sinusoidal" or "learned". */
    public String mode;

    // ================================================================
    // E12 — Embeddings
    // ================================================================

    /**
     * Builds a sinusoidal positional encoding table.
     *
     * <p>PE(pos, 2i) = sin(pos / 10000^(2i/dim))</p>
     * <p>PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))</p>
     *
     * @param maxLen maximum sequence length
     * @param dim    embedding dimension
     * @return NDArray of shape [maxLen, dim]
     */
    public static NDArray buildSinusoidalTable(int maxLen, int dim) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    /**
     * Creates a PositionalEncoding layer.
     *
     * @param mode   "sinusoidal" or "learned"
     * @param maxLen maximum sequence length
     * @param dim    embedding dimension
     */
    public PositionalEncoding(String mode, int maxLen, int dim) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    /**
     * Adds positional encoding to token embeddings.
     *
     * @param tokenEmbeddings tensor of shape [seqLen, dim]
     * @return tensor of shape [seqLen, dim] with position info added
     */
    public Tensor forward(Tensor tokenEmbeddings) {
        throw new UnsupportedOperationException("TODO: E12");
    }

    /**
     * Returns trainable parameters (empty for sinusoidal, one weight for learned).
     */
    public List<Tensor> parameters() {
        throw new UnsupportedOperationException("TODO: E12");
    }
}
