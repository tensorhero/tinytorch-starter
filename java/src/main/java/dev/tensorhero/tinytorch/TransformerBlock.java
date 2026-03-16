package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TransformerBlock — LayerNorm, MLP, and Pre-norm Transformer Block.
 *
 * <p>Part 1: LayerNorm — layer normalization with learnable gamma/beta.
 * Part 2: MLP — two-layer feed-forward network with GELU activation.
 * Part 3: TransformerBlock — Pre-norm architecture with residual connections.</p>
 */
public class TransformerBlock {

    // ================================================================
    // Part 1: LayerNorm
    // ================================================================

    /**
     * Layer Normalization.
     *
     * <p>Normalizes the input along the last dimension:
     * output = gamma * (x - mean) / sqrt(var + eps) + beta</p>
     */
    public static class LayerNorm extends Layer {

        public Tensor gamma;
        public Tensor beta;
        public float eps;
        public int dim;

        public LayerNorm(int dim) {
            this(dim, 1e-5f);
        }

        public LayerNorm(int dim, float eps) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public List<Tensor> parameters() {
            return Arrays.asList(gamma, beta);
        }
    }

    // ================================================================
    // Part 2: MLP
    // ================================================================

    /**
     * MLP (Multi-Layer Perceptron) with 4x expansion.
     *
     * <p>Structure: fc1 (D → 4D) → GELU → fc2 (4D → D)</p>
     */
    public static class MLP extends Layer {

        public Linear fc1, fc2;

        public MLP(int dim) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public List<Tensor> parameters() {
            List<Tensor> params = new ArrayList<>();
            params.addAll(fc1.parameters());
            params.addAll(fc2.parameters());
            return params;
        }

        @Override
        public List<Layer> children() {
            List<Layer> kids = new ArrayList<>();
            kids.add(fc1);
            kids.add(fc2);
            return kids;
        }
    }

    // ================================================================
    // Part 3: TransformerBlock
    // ================================================================

    /**
     * Pre-norm Transformer Block.
     *
     * <p>Architecture:
     * x = x + attn(ln1(x))
     * x = x + mlp(ln2(x))</p>
     */
    public static class Block extends Layer {

        public LayerNorm ln1, ln2;
        public Attention.MultiHeadAttention attn;
        public MLP mlp;

        public Block(int dim, int numHeads) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("TODO: E14");
        }

        @Override
        public List<Tensor> parameters() {
            List<Tensor> params = new ArrayList<>();
            params.addAll(ln1.parameters());
            params.addAll(attn.parameters());
            params.addAll(ln2.parameters());
            params.addAll(mlp.parameters());
            return params;
        }

        @Override
        public List<Layer> children() {
            List<Layer> kids = new ArrayList<>();
            kids.add(ln1);
            kids.add(attn);
            kids.add(ln2);
            kids.add(mlp);
            return kids;
        }
    }
}
