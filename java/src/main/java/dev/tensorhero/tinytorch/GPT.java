package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT — complete GPT language model with autoregressive generation.
 *
 * <p>Stacks Embedding, PositionalEncoding, TransformerBlock layers,
 * a final LayerNorm, and a linear head to produce token logits.
 * Includes a generate method for autoregressive text generation.</p>
 */
public class GPT extends Layer {

    public Embedding tokenEmb;
    public PositionalEncoding posEnc;
    public TransformerBlock.Block[] blocks;
    public TransformerBlock.LayerNorm lnFinal;
    public Linear lmHead;
    public int vocabSize;
    public int dim;
    public int numHeads;
    public int numLayers;
    public int maxLen;

    // ================================================================
    // E15 — GPT & Generate
    // ================================================================

    /**
     * Creates a GPT model.
     *
     * @param vocabSize vocabulary size
     * @param dim       model dimension (embedding size)
     * @param numHeads  number of attention heads per block
     * @param numLayers number of transformer blocks
     * @param maxLen    maximum sequence length
     */
    public GPT(int vocabSize, int dim, int numHeads, int numLayers, int maxLen) {
        throw new UnsupportedOperationException("TODO: E15");
    }

    /**
     * Forward pass: token IDs → logits.
     *
     * @param tokenIds array of token IDs
     * @return logits tensor of shape [seqLen, vocabSize]
     */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("TODO: E15");
    }

    /**
     * Forward pass taking raw int[] token IDs.
     *
     * @param tokenIds array of token IDs
     * @return logits tensor of shape [seqLen, vocabSize]
     */
    public Tensor forward(int[] tokenIds) {
        throw new UnsupportedOperationException("TODO: E15");
    }

    /**
     * Autoregressive text generation.
     *
     * @param tokenizer   the tokenizer for encode/decode
     * @param prompt      the starting text
     * @param maxTokens   number of tokens to generate
     * @param temperature sampling temperature (0 = greedy)
     * @return the generated text (prompt + generated)
     */
    public String generate(Tokenizer.CharTokenizer tokenizer, String prompt,
                           int maxTokens, float temperature) {
        throw new UnsupportedOperationException("TODO: E15");
    }

    @Override
    public List<Tensor> parameters() {
        throw new UnsupportedOperationException("TODO: E15");
    }

    @Override
    public List<Layer> children() {
        throw new UnsupportedOperationException("TODO: E15");
    }
}
