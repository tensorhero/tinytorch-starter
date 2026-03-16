package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Attention — Scaled Dot-Product Attention and Multi-Head Attention.
 *
 * <p>Part 1: Static utility methods for causal mask and scaled dot-product attention.
 * Part 2: MultiHeadAttention layer with linear projections and head splitting.</p>
 */
public class Attention {

    // ================================================================
    // Part 1: Causal Mask & Scaled Dot-Product Attention
    // ================================================================

    /**
     * Creates a causal mask of shape [seqLen, seqLen].
     * Mask value is 1.0 where the position should be masked (upper triangle),
     * and 0.0 where the position is visible (lower triangle + diagonal).
     *
     * @param seqLen sequence length
     * @return NDArray mask of shape [seqLen, seqLen]
     */
    public static NDArray createCausalMask(int seqLen) {
        throw new UnsupportedOperationException("TODO: E13");
    }

    /**
     * Scaled Dot-Product Attention.
     *
     * <p>Computes attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V,
     * with optional masking.</p>
     *
     * @param Q    query tensor, shape [..., T, d_k]
     * @param K    key tensor, shape [..., T, d_k]
     * @param V    value tensor, shape [..., T, d_k]
     * @param mask optional mask (NDArray), 1.0 at positions to mask, null for no mask
     * @return attention output, same shape as V
     */
    public static Tensor scaledDotProductAttention(Tensor Q, Tensor K, Tensor V, NDArray mask) {
        throw new UnsupportedOperationException("TODO: E13");
    }

    // ================================================================
    // Part 2: Multi-Head Attention
    // ================================================================

    /**
     * Multi-Head Attention layer.
     *
     * <p>Splits the input into multiple heads, applies scaled dot-product
     * attention independently per head, then merges and projects the output.</p>
     */
    public static class MultiHeadAttention extends Layer {

        public Linear qProj, kProj, vProj, outProj;
        public int numHeads;
        public int headDim;

        /**
         * Creates a Multi-Head Attention layer.
         *
         * @param embedDim total embedding dimension (must be divisible by numHeads)
         * @param numHeads number of attention heads
         */
        public MultiHeadAttention(int embedDim, int numHeads) {
            throw new UnsupportedOperationException("TODO: E13");
        }

        /**
         * Forward pass: project → split heads → attention → merge → project.
         *
         * @param input tensor of shape [B, T, D]
         * @return output tensor of shape [B, T, D]
         */
        @Override
        public Tensor forward(Tensor input) {
            throw new UnsupportedOperationException("TODO: E13");
        }

        @Override
        public List<Tensor> parameters() {
            throw new UnsupportedOperationException("TODO: E13");
        }

        @Override
        public List<Layer> children() {
            throw new UnsupportedOperationException("TODO: E13");
        }
    }
}
