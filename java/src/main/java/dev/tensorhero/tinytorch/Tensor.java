package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;
import java.util.List;
import java.util.function.Supplier;

/**
 * Tensor — the core data structure of TinyTorch.
 *
 * <p>Wraps an {@link NDArray} from TinyNum and adds gradient metadata
 * ({@code requiresGrad}, {@code grad}, {@code gradFn}) for automatic
 * differentiation. In E01-E04 these fields remain dormant.</p>
 */
public class Tensor {

    /** The underlying N-dimensional array (from TinyNum). */
    public NDArray data;

    /** Whether this tensor requires gradient computation. Default: false. */
    public boolean requiresGrad;

    /** Accumulated gradient (populated after backward). Default: null. */
    public Tensor grad;

    /** The Function that created this tensor (for autograd graph). Default: null. */
    public Function gradFn;

    // ================================================================
    // E01 — Tensor Class
    // ================================================================

    // --- Factory methods ---

    /**
     * Creates a Tensor wrapping an existing NDArray.
     *
     * @param data the NDArray to wrap
     * @return a new Tensor
     */
    public static Tensor fromNDArray(NDArray data) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Creates a Tensor from a flat data array with the given shape.
     *
     * @param data  the flat data array
     * @param shape the desired shape
     * @return a new Tensor
     */
    public static Tensor fromArray(float[] data, int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Creates a zero-filled Tensor with the given shape.
     *
     * @param shape the desired shape
     * @return a new Tensor of zeros
     */
    public static Tensor zeros(int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Creates a one-filled Tensor with the given shape.
     *
     * @param shape the desired shape
     * @return a new Tensor of ones
     */
    public static Tensor ones(int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Creates a Tensor filled with the given value.
     *
     * @param value the fill value
     * @param shape the desired shape
     * @return a new Tensor
     */
    public static Tensor full(float value, int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Creates a Tensor with standard normal random values (mean=0, std=1).
     *
     * @param shape the desired shape
     * @return a new Tensor of random values
     */
    public static Tensor randn(int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    // --- Element-wise operations (delegate to NDArray) ---

    /**
     * Element-wise addition.
     *
     * @param other the tensor to add
     * @return a new Tensor with the result
     */
    public Tensor add(Tensor other) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Element-wise subtraction.
     *
     * @param other the tensor to subtract
     * @return a new Tensor with the result
     */
    public Tensor sub(Tensor other) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Element-wise multiplication.
     *
     * @param other the tensor to multiply
     * @return a new Tensor with the result
     */
    public Tensor mul(Tensor other) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Element-wise division.
     *
     * @param other the tensor to divide by
     * @return a new Tensor with the result
     */
    public Tensor div(Tensor other) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Matrix multiplication.
     *
     * @param other the tensor to multiply with
     * @return a new Tensor with the result
     */
    public Tensor matMul(Tensor other) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    // --- Reduction operations ---

    /**
     * Sum along the given axis.
     *
     * @param axis     the axis to sum along
     * @param keepDims whether to keep the reduced dimension
     * @return a new Tensor with the result
     */
    public Tensor sum(int axis, boolean keepDims) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Mean along the given axis.
     *
     * @param axis     the axis to average along
     * @param keepDims whether to keep the reduced dimension
     * @return a new Tensor with the result
     */
    public Tensor mean(int axis, boolean keepDims) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    // --- Shape operations ---

    /**
     * Returns a new Tensor with the given shape (same data, different view).
     *
     * @param shape the new shape
     * @return a reshaped Tensor
     */
    public Tensor reshape(int... shape) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Returns a transposed Tensor with axes permuted.
     *
     * @param axes the new axis order
     * @return a transposed Tensor
     */
    public Tensor transpose(int... axes) {
        throw new UnsupportedOperationException("TODO: E01");
    }

    // --- Properties ---

    /**
     * Returns the shape of this tensor.
     *
     * @return a copy of the shape array
     */
    public int[] shape() {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the number of dimensions
     */
    public int ndim() {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Returns the total number of elements.
     *
     * @return the total element count
     */
    public int size() {
        throw new UnsupportedOperationException("TODO: E01");
    }

    /**
     * Returns a string representation of this tensor.
     *
     * @return the string representation (delegates to NDArray)
     */
    @Override
    public String toString() {
        throw new UnsupportedOperationException("TODO: E01");
    }

    // ================================================================
    // E05 — Computation Graph
    // ================================================================
    // When requiresGrad is true on any input, add(), sub(), mul(), div()
    // should create the corresponding Backward function, call forward(),
    // and set result.requiresGrad = true and result.gradFn = fn.
    // Otherwise, use the fast path (direct NDArray delegation from E01).

    // ================================================================
    // E06 — More Backward Ops
    // ================================================================
    // Extend graph recording to: matMul, sum, mean, reshape, transpose.
    // Add new methods: exp(), log() (with ExpBackward, LogBackward).
    // Modify Activations: relu, sigmoid, tanh, gelu → use corresponding Backward.
    // Modify Losses: crossEntropy → CrossEntropyBackward; mse → use Tensor ops.
    // Modify Dropout: forward → DropoutBackward.

    /**
     * Element-wise exponential: y = exp(x).
     *
     * @return a new Tensor with the result
     */
    public Tensor exp() {
        throw new UnsupportedOperationException("TODO: E06");
    }

    /**
     * Element-wise natural log: y = log(x).
     *
     * @return a new Tensor with the result
     */
    public Tensor log() {
        throw new UnsupportedOperationException("TODO: E06");
    }

    // ================================================================
    // E07 — Backpropagation
    // ================================================================
    // Implement backward(), topologicalSort(), reduceBroadcastGrad(), and noGrad().
    // Also add gradEnabled check to all graph-recording methods (add, sub, mul, etc.).

    /** Global switch to enable/disable gradient computation. */
    public static boolean gradEnabled = true;

    /**
     * Topological sort of the computation graph rooted at this tensor.
     * Uses DFS post-order: leaves first, root last.
     *
     * @param root the root tensor (typically the loss)
     * @return list in topological order (leaves → root)
     */
    public static List<Tensor> topologicalSort(Tensor root) {
        throw new UnsupportedOperationException("TODO: E07");
    }

    /**
     * Reduce a gradient that was broadcast during forward pass back to the target shape.
     * Sums along axes that were broadcast (leading extra dims + size-1 dims).
     *
     * @param grad        the gradient to reduce
     * @param targetShape the original shape to reduce to
     * @return the reduced gradient matching targetShape
     */
    public static NDArray reduceBroadcastGrad(NDArray grad, int[] targetShape) {
        throw new UnsupportedOperationException("TODO: E07");
    }

    /**
     * Backpropagation: compute gradients for all tensors in the computation graph.
     * Must be called on a scalar tensor (size == 1).
     * Sets .grad on each tensor with requiresGrad == true.
     */
    public void backward() {
        throw new UnsupportedOperationException("TODO: E07");
    }

    /**
     * Execute a block with gradient computation disabled.
     * Restores the previous gradEnabled state even if an exception occurs.
     *
     * @param block the computation to run without gradient tracking
     * @param <T>   the return type
     * @return the result of block.get()
     */
    public static <T> T noGrad(Supplier<T> block) {
        throw new UnsupportedOperationException("TODO: E07");
    }
}
