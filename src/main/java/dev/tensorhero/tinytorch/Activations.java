package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Activations — element-wise nonlinear activation functions.
 *
 * <p>All methods are static, accept a {@link Tensor}, and return a new
 * {@link Tensor}. No learnable parameters. Forward-only in E02;
 * backward implementations will be added in E06.</p>
 */
public class Activations {

    // ================================================================
    // E02 — Activations
    // ================================================================

    /**
     * ReLU activation: f(x) = max(0, x).
     *
     * @param x the input tensor
     * @return a new Tensor with ReLU applied element-wise
     */
    public static Tensor relu(Tensor x) {
        throw new UnsupportedOperationException("TODO: E02");
    }

    /**
     * Sigmoid activation: f(x) = 1 / (1 + exp(-x)).
     *
     * @param x the input tensor
     * @return a new Tensor with sigmoid applied element-wise
     */
    public static Tensor sigmoid(Tensor x) {
        throw new UnsupportedOperationException("TODO: E02");
    }

    /**
     * Tanh activation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
     *
     * @param x the input tensor
     * @return a new Tensor with tanh applied element-wise
     */
    public static Tensor tanh(Tensor x) {
        throw new UnsupportedOperationException("TODO: E02");
    }

    /**
     * GELU activation (tanh approximation):
     * f(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
     *
     * @param x the input tensor
     * @return a new Tensor with GELU applied element-wise
     */
    public static Tensor gelu(Tensor x) {
        throw new UnsupportedOperationException("TODO: E02");
    }

    /**
     * Softmax activation: f(x_i) = exp(x_i - max) / sum(exp(x_j - max)).
     *
     * <p>Numerically stable: subtracts the max before computing exp
     * to prevent overflow.</p>
     *
     * @param x    the input tensor
     * @param axis the axis along which to compute softmax
     * @return a new Tensor with softmax applied along the given axis
     */
    public static Tensor softmax(Tensor x, int axis) {
        throw new UnsupportedOperationException("TODO: E02");
    }

    // ================================================================
    // E06 — More Backward Ops
    // ================================================================
    // When x.requiresGrad is true, relu(), sigmoid(), tanh(), and gelu()
    // should create the corresponding Backward function (ReLUBackward,
    // SigmoidBackward, TanhBackward, GELUBackward), call fn.forward(),
    // and set result.requiresGrad = true, result.gradFn = fn.
    // softmax does NOT need backward here (CrossEntropy uses fusion).
}
