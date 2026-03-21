package dev.tensorhero.tinynum;

/**
 * N-dimensional array — the core data structure of tinynum.
 *
 * <p>Internally stores data in a flat {@code float[]} with shape and strides
 * metadata, enabling zero-copy views for reshape, transpose, and slice.</p>
 */
public class NDArray {

    // Shared RNG — call manualSeed() for reproducible results.
    private static java.util.Random globalRng = new java.util.Random();

    /** Seed the global RNG for reproducible random operations. */
    public static void manualSeed(long seed) { globalRng = new java.util.Random(seed); }

    float[] data;       // flat storage
    int[] shape;        // e.g. [2, 3, 4]
    int[] strides;      // e.g. [12, 4, 1] (row-major)
    int offset;         // for views/slices

    // Private helper: compute row-major strides from shape.
    // E02 exposes this as a public API; here it's used internally by E01 constructors.
    private static int[] rowMajorStrides(int[] shape) {
        int[] strides = new int[shape.length];
        if (shape.length > 0) {
            strides[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }

    // ================================================================
    // E01 — Storage & Shape
    // ================================================================

    /**
     * Creates an NDArray from a flat data array with the given shape.
     *
     * @param data the flat data array
     * @param shape the desired shape (e.g. 2, 3 for a 2×3 matrix)
     * @return a new NDArray
     * @throws IllegalArgumentException if data.length != product of shape
     */
    public static NDArray fromArray(float[] data, int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        if (data.length != size) {
            throw new IllegalArgumentException(
                "data length " + data.length + " does not match shape product " + size);
        }
        NDArray arr = new NDArray();
        arr.data = data.clone();
        arr.shape = shape.clone();
        arr.strides = rowMajorStrides(shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates a zero-filled NDArray with the given shape. */
    public static NDArray zeros(int... shape) {
        return full(0.0f, shape);
    }

    /** Creates a one-filled NDArray with the given shape. */
    public static NDArray ones(int... shape) {
        return full(1.0f, shape);
    }

    /** Creates an NDArray filled with {@code value}. */
    public static NDArray full(float value, int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        float[] data = new float[size];
        if (value != 0.0f) {
            java.util.Arrays.fill(data, value);
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = shape.clone();
        arr.strides = rowMajorStrides(shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates a zero-filled NDArray with the same shape as {@code other}. */
    public static NDArray zerosLike(NDArray other) {
        return zeros(other.shape());
    }

    /** Creates a one-filled NDArray with the same shape as {@code other}. */
    public static NDArray onesLike(NDArray other) {
        return ones(other.shape());
    }

    /** Returns the total number of elements. */
    public int size() {
        int s = 1;
        for (int d : shape) s *= d;
        return s;
    }

    /** Returns the number of dimensions. */
    public int ndim() {
        return shape.length;
    }

    /** Returns a copy of the shape array. */
    public int[] shape() {
        return shape.clone();
    }

    /** Pretty-prints the NDArray: e.g. {@code [[1.0, 2.0], [3.0, 4.0]]}. */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        formatRecursive(sb, 0, offset);
        return sb.toString();
    }

    private void formatRecursive(StringBuilder sb, int dim, int flatOffset) {
        if (dim == shape.length - 1) {
            sb.append('[');
            for (int i = 0; i < shape[dim]; i++) {
                if (i > 0) sb.append(", ");
                sb.append(Float.toString(data[flatOffset + i * strides[dim]]));
            }
            sb.append(']');
        } else {
            sb.append('[');
            for (int i = 0; i < shape[dim]; i++) {
                if (i > 0) sb.append(", ");
                formatRecursive(sb, dim + 1, flatOffset + i * strides[dim]);
            }
            sb.append(']');
        }
    }

    // ================================================================
    // E02 — Strides & Indexing
    // ================================================================

    /**
     * Computes row-major strides for the given shape.
     * <p>Example: shape [3, 4, 5] → strides [20, 5, 1]</p>
     */
    public static int[] computeStrides(int[] shape) {
        return rowMajorStrides(shape);
    }

    /**
     * Gets the element at the given multi-dimensional indices.
     * <p>Uses: {@code physicalIndex = offset + Σ(index[i] × stride[i])}</p>
     */
    public float get(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "expected " + shape.length + " indices but got " + indices.length);
        }
        int idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * strides[i];
        }
        return data[idx];
    }

    /** Sets the element at the given multi-dimensional indices. */
    public void set(float value, int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "expected " + shape.length + " indices but got " + indices.length);
        }
        int idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * strides[i];
        }
        data[idx] = value;
    }

    /** Returns true if strides form a standard row-major contiguous layout. */
    public boolean isContiguous() {
        int[] expected = rowMajorStrides(shape);
        return java.util.Arrays.equals(strides, expected);
    }

    // ================================================================
    // E03 — Reshape
    // ================================================================

    /**
     * Returns a view with a new shape (zero-copy when contiguous).
     * Supports -1 for one dimension to auto-infer its size.
     */
    public NDArray reshape(int... newShape) {
        // Resolve -1
        int negIdx = -1;
        int knownProduct = 1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (negIdx != -1) {
                    throw new IllegalArgumentException("only one -1 allowed in reshape");
                }
                negIdx = i;
            } else {
                knownProduct *= newShape[i];
            }
        }
        int[] resolved = newShape.clone();
        int sz = size();
        if (negIdx != -1) {
            if (sz % knownProduct != 0) {
                throw new IllegalArgumentException(
                    "cannot reshape array of size " + sz + " into shape with known product " + knownProduct);
            }
            resolved[negIdx] = sz / knownProduct;
        }
        // Validate total size
        int newSize = 1;
        for (int d : resolved) newSize *= d;
        if (newSize != sz) {
            throw new IllegalArgumentException(
                "cannot reshape array of size " + sz + " into shape of size " + newSize);
        }
        // If non-contiguous, duplicate first
        NDArray src = isContiguous() ? this : this.duplicate();
        NDArray view = new NDArray();
        view.data = src.data;
        view.shape = resolved;
        view.strides = rowMajorStrides(resolved);
        view.offset = src.offset;
        return view;
    }

    /** Flattens to a 1-D array. Equivalent to {@code reshape(-1)}. */
    public NDArray flatten() {
        return reshape(-1);
    }

    /** Returns a deep copy (always contiguous). */
    public NDArray duplicate() {
        float[] newData = new float[size()];
        int[] idx = new int[shape.length];
        for (int i = 0; i < newData.length; i++) {
            newData[i] = get(idx);
            // Increment multi-dimensional index (row-major order)
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return fromArray(newData, shape.clone());
    }

    // ================================================================
    // E04 — Transpose
    // ================================================================

    /** 2-D transpose: swaps axis 0 and axis 1 (zero-copy). */
    public NDArray transpose() {
        if (shape.length != 2) {
            throw new IllegalArgumentException("transpose() with no args requires a 2-D array");
        }
        return transpose(1, 0);
    }

    /** N-D transpose: rearranges axes according to the given permutation (zero-copy). */
    public NDArray transpose(int... axes) {
        int ndim = shape.length;
        if (axes.length != ndim) {
            throw new IllegalArgumentException("axes length must match ndim");
        }
        boolean[] seen = new boolean[ndim];
        for (int a : axes) {
            if (a < 0 || a >= ndim || seen[a]) {
                throw new IllegalArgumentException("invalid axes permutation");
            }
            seen[a] = true;
        }
        int[] newShape = new int[ndim];
        int[] newStrides = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            newShape[i] = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = this.offset;
        return result;
    }

    /** Swaps two axes (zero-copy). */
    public NDArray swapAxes(int axis1, int axis2) {
        int ndim = shape.length;
        if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
            throw new IllegalArgumentException("axis out of range");
        }
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) {
            axes[i] = i;
        }
        axes[axis1] = axis2;
        axes[axis2] = axis1;
        return transpose(axes);
    }

    // ================================================================
    // E05 — Unary Math
    // ================================================================

    // Internal iterator: applies fn to every element, returns a new contiguous array.
    private NDArray applyUnary(java.util.function.DoubleUnaryOperator fn) {
        int n = size();
        float[] result = new float[n];
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            result[i] = (float) fn.applyAsDouble(get(idx));
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return fromArray(result, shape.clone());
    }

    /** Returns {@code -x} element-wise. */
    public NDArray neg() {
        return applyUnary(x -> -x);
    }

    /** Returns {@code |x|} element-wise. */
    public NDArray abs() {
        return applyUnary(x -> Math.abs(x));
    }

    /** Returns {@code e^x} element-wise. */
    public NDArray exp() {
        return applyUnary(x -> (float) Math.exp(x));
    }

    /** Returns {@code ln(x)} element-wise. */
    public NDArray log() {
        return applyUnary(x -> (float) Math.log(x));
    }

    /** Returns {@code √x} element-wise. */
    public NDArray sqrt() {
        return applyUnary(x -> (float) Math.sqrt(x));
    }

    /** Returns {@code x²} element-wise. */
    public NDArray square() {
        return applyUnary(x -> x * x);
    }

    /** Returns {@code tanh(x)} element-wise. */
    public NDArray tanh() {
        return applyUnary(x -> (float) Math.tanh(x));
    }

    /** Returns {@code sin(x)} element-wise. */
    public NDArray sin() {
        return applyUnary(x -> (float) Math.sin(x));
    }

    /** Returns {@code cos(x)} element-wise. */
    public NDArray cos() {
        return applyUnary(x -> (float) Math.cos(x));
    }

    /** Returns {@code sgn(x)} element-wise. */
    public NDArray sign() {
        return applyUnary(x -> Math.signum(x));
    }

    /** Returns rounded values element-wise. */
    public NDArray round() {
        return applyUnary(x -> (float) Math.round(x));
    }

    /** Clips values to {@code [min, max]} element-wise. */
    public NDArray clip(float min, float max) {
        return applyUnary(x -> Math.max(min, Math.min(max, x)));
    }

    /** Returns {@code x^p} element-wise. */
    public NDArray pow(float p) {
        return applyUnary(x -> (float) Math.pow(x, p));
    }

    // ================================================================
    // E06 — Binary Ops & Comparisons (same shape)
    // ================================================================

    // Internal iterator: applies fn to every pair of elements from this and other.
    // Automatically broadcasts operands to a common shape (E07).
    private NDArray applyBinary(NDArray other, java.util.function.DoubleBinaryOperator fn) {
        int[] outShape = broadcastShapes(this.shape, other.shape);
        NDArray a = this.broadcastTo(outShape);
        NDArray b = other.broadcastTo(outShape);
        int n = 1;
        for (int s : outShape) n *= s;
        float[] result = new float[n];
        int[] idx = new int[outShape.length];
        for (int i = 0; i < n; i++) {
            result[i] = (float) fn.applyAsDouble(a.get(idx), b.get(idx));
            for (int d = outShape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < outShape[d]) break;
                idx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    // --- Arithmetic (NDArray) ---

    /** Element-wise addition. */
    public NDArray add(NDArray other) {
        return applyBinary(other, (a, b) -> a + b);
    }

    /** Element-wise subtraction. */
    public NDArray sub(NDArray other) {
        return applyBinary(other, (a, b) -> a - b);
    }

    /** Element-wise multiplication. */
    public NDArray mul(NDArray other) {
        return applyBinary(other, (a, b) -> a * b);
    }

    /** Element-wise division. */
    public NDArray div(NDArray other) {
        return applyBinary(other, (a, b) -> a / b);
    }

    /** Element-wise power: {@code x^y}. */
    public NDArray pow(NDArray other) {
        return applyBinary(other, (a, b) -> (float) Math.pow(a, b));
    }

    /** Element-wise maximum: {@code max(x, y)}. */
    public NDArray maximum(NDArray other) {
        return applyBinary(other, Math::max);
    }

    // --- Arithmetic (scalar) ---

    /** Adds a scalar to every element. */
    public NDArray add(float scalar) {
        return applyUnary(x -> x + scalar);
    }

    /** Subtracts a scalar from every element. */
    public NDArray sub(float scalar) {
        return applyUnary(x -> x - scalar);
    }

    /** Multiplies every element by a scalar. */
    public NDArray mul(float scalar) {
        return applyUnary(x -> x * scalar);
    }

    /** Divides every element by a scalar. */
    public NDArray div(float scalar) {
        return applyUnary(x -> x / scalar);
    }

    // --- Comparisons (NDArray) — returns 1.0f / 0.0f ---

    /** Element-wise equal: returns 1.0 where {@code x == y}. */
    public NDArray eq(NDArray other) {
        return applyBinary(other, (a, b) -> a == b ? 1.0 : 0.0);
    }

    /** Element-wise not-equal: returns 1.0 where {@code x != y}. */
    public NDArray neq(NDArray other) {
        return applyBinary(other, (a, b) -> a != b ? 1.0 : 0.0);
    }

    /** Element-wise greater-than: returns 1.0 where {@code x > y}. */
    public NDArray gt(NDArray other) {
        return applyBinary(other, (a, b) -> a > b ? 1.0 : 0.0);
    }

    /** Element-wise greater-than-or-equal: returns 1.0 where {@code x >= y}. */
    public NDArray gte(NDArray other) {
        return applyBinary(other, (a, b) -> a >= b ? 1.0 : 0.0);
    }

    /** Element-wise less-than: returns 1.0 where {@code x < y}. */
    public NDArray lt(NDArray other) {
        return applyBinary(other, (a, b) -> a < b ? 1.0 : 0.0);
    }

    /** Element-wise less-than-or-equal: returns 1.0 where {@code x <= y}. */
    public NDArray lte(NDArray other) {
        return applyBinary(other, (a, b) -> a <= b ? 1.0 : 0.0);
    }

    // --- Comparisons (scalar) ---

    public NDArray eq(float scalar) {
        return applyUnary(x -> x == scalar ? 1.0 : 0.0);
    }

    public NDArray neq(float scalar) {
        return applyUnary(x -> x != scalar ? 1.0 : 0.0);
    }

    public NDArray gt(float scalar) {
        return applyUnary(x -> x > scalar ? 1.0 : 0.0);
    }

    public NDArray gte(float scalar) {
        return applyUnary(x -> x >= scalar ? 1.0 : 0.0);
    }

    public NDArray lt(float scalar) {
        return applyUnary(x -> x < scalar ? 1.0 : 0.0);
    }

    public NDArray lte(float scalar) {
        return applyUnary(x -> x <= scalar ? 1.0 : 0.0);
    }

    // ================================================================
    // E07 — Broadcasting
    // ================================================================

    /**
     * Computes the broadcast-compatible output shape for two input shapes.
     * <p>Example: [3, 1] + [1, 4] → [3, 4]</p>
     *
     * @throws IllegalArgumentException if shapes are not broadcast-compatible
     */
    public static int[] broadcastShapes(int[] shapeA, int[] shapeB) {
        int ndim = Math.max(shapeA.length, shapeB.length);
        int[] result = new int[ndim];
        for (int i = ndim - 1; i >= 0; i--) {
            int a = (i - (ndim - shapeA.length) >= 0) ? shapeA[i - (ndim - shapeA.length)] : 1;
            int b = (i - (ndim - shapeB.length) >= 0) ? shapeB[i - (ndim - shapeB.length)] : 1;
            if (a == b) {
                result[i] = a;
            } else if (a == 1) {
                result[i] = b;
            } else if (b == 1) {
                result[i] = a;
            } else {
                throw new IllegalArgumentException(
                    "Shapes not broadcastable: " + java.util.Arrays.toString(shapeA) +
                    " vs " + java.util.Arrays.toString(shapeB));
            }
        }
        return result;
    }

    /**
     * Returns a view broadcast to the target shape (zero-copy, stride=0 trick).
     */
    public NDArray broadcastTo(int... targetShape) {
        int ndim = targetShape.length;
        if (ndim < this.shape.length) {
            throw new IllegalArgumentException("target ndim must be >= source ndim");
        }
        int[] newStrides = new int[ndim];
        for (int i = ndim - 1; i >= 0; i--) {
            int srcIdx = i - (ndim - this.shape.length);
            int origDim = (srcIdx >= 0) ? this.shape[srcIdx] : 1;
            if (origDim == targetShape[i]) {
                newStrides[i] = (srcIdx >= 0) ? this.strides[srcIdx] : 0;
            } else if (origDim == 1) {
                newStrides[i] = 0;
            } else {
                throw new IllegalArgumentException(
                    "Cannot broadcast shape " + java.util.Arrays.toString(this.shape) +
                    " to " + java.util.Arrays.toString(targetShape));
            }
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = targetShape.clone();
        result.strides = newStrides;
        result.offset = this.offset;
        return result;
    }

    // ================================================================
    // E08 — Reduction: Sum & Mean
    // ================================================================

    /** Returns the sum of all elements. */
    public float sum() {
        int n = size();
        float total = 0;
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            total += get(idx);
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return total;
    }

    /** Returns the mean of all elements. */
    public float mean() {
        return sum() / size();
    }

    /** Sum along a single axis. */
    public NDArray sum(int axis, boolean keepDims) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for ndim " + shape.length);
        }
        // Build output shape
        int[] outShape;
        if (keepDims) {
            outShape = shape.clone();
            outShape[ax] = 1;
        } else {
            outShape = new int[shape.length - 1];
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i != ax) outShape[j++] = shape[i];
            }
        }
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];
        // Iterate over output positions
        int[] outIdx = new int[outShape.length];
        for (int i = 0; i < outSize; i++) {
            // Map outIdx → inIdx (insert reduction axis)
            int[] inIdx = new int[shape.length];
            if (keepDims) {
                System.arraycopy(outIdx, 0, inIdx, 0, outIdx.length);
            } else {
                for (int d = 0, j = 0; d < shape.length; d++) {
                    if (d != ax) inIdx[d] = outIdx[j++];
                }
            }
            float total = 0;
            for (int j = 0; j < shape[ax]; j++) {
                inIdx[ax] = j;
                total += get(inIdx);
            }
            result[i] = total;
            // Advance outIdx
            for (int d = outShape.length - 1; d >= 0; d--) {
                outIdx[d]++;
                if (outIdx[d] < outShape[d]) break;
                outIdx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Mean along a single axis. */
    public NDArray mean(int axis, boolean keepDims) {
        int ax = axis < 0 ? shape.length + axis : axis;
        NDArray s = sum(axis, keepDims);
        int count = shape[ax];
        return s.div(count);
    }

    /** Sum along multiple axes. */
    public NDArray sum(int[] axes, boolean keepDims) {
        // Normalize and sort descending to avoid axis shift
        int[] sorted = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            sorted[i] = axes[i] < 0 ? shape.length + axes[i] : axes[i];
        }
        java.util.Arrays.sort(sorted);
        // Reverse to descending
        for (int i = 0; i < sorted.length / 2; i++) {
            int tmp = sorted[i];
            sorted[i] = sorted[sorted.length - 1 - i];
            sorted[sorted.length - 1 - i] = tmp;
        }
        NDArray result = this;
        for (int ax : sorted) {
            result = result.sum(ax, keepDims);
        }
        return result;
    }

    // ================================================================
    // E09 — Reduction: Max, Var & friends
    // ================================================================

    // Internal: generalized axis reduction that tracks both value and index.
    // combine(bestVal, bestIdx, currentVal, currentIdx) -> new float[]{bestVal, bestIdx}
    private NDArray reduceAxis(int axis, boolean keepDims, float initVal,
                               java.util.function.DoubleBinaryOperator combine) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for ndim " + shape.length);
        }
        int[] outShape;
        if (keepDims) {
            outShape = shape.clone();
            outShape[ax] = 1;
        } else {
            outShape = new int[shape.length - 1];
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i != ax) outShape[j++] = shape[i];
            }
        }
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];
        int[] outIdx = new int[outShape.length];
        for (int i = 0; i < outSize; i++) {
            int[] inIdx = new int[shape.length];
            if (keepDims) {
                System.arraycopy(outIdx, 0, inIdx, 0, outIdx.length);
            } else {
                for (int d = 0, j = 0; d < shape.length; d++) {
                    if (d != ax) inIdx[d] = outIdx[j++];
                }
            }
            inIdx[ax] = 0;
            float acc = (initVal == Float.MIN_VALUE) ? get(inIdx) : initVal;
            int start = (initVal == Float.MIN_VALUE) ? 1 : 0;
            for (int j = start; j < shape[ax]; j++) {
                inIdx[ax] = j;
                acc = (float) combine.applyAsDouble(acc, get(inIdx));
            }
            result[i] = acc;
            for (int d = outShape.length - 1; d >= 0; d--) {
                outIdx[d]++;
                if (outIdx[d] < outShape[d]) break;
                outIdx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Max along an axis. */
    public NDArray max(int axis, boolean keepDims) {
        return reduceAxis(axis, keepDims, Float.MIN_VALUE, Math::max);
    }

    /** Min along an axis. */
    public NDArray min(int axis, boolean keepDims) {
        return reduceAxis(axis, keepDims, Float.MIN_VALUE, Math::min);
    }

    /** Index of the maximum value along an axis. */
    public NDArray argmax(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for ndim " + shape.length);
        }
        int[] outShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != ax) outShape[j++] = shape[i];
        }
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];
        int[] outIdx = new int[outShape.length];
        for (int i = 0; i < outSize; i++) {
            int[] inIdx = new int[shape.length];
            for (int d = 0, j = 0; d < shape.length; d++) {
                if (d != ax) inIdx[d] = outIdx[j++];
            }
            inIdx[ax] = 0;
            float bestVal = get(inIdx);
            int bestIdx = 0;
            for (int j = 1; j < shape[ax]; j++) {
                inIdx[ax] = j;
                float val = get(inIdx);
                if (val > bestVal) {
                    bestVal = val;
                    bestIdx = j;
                }
            }
            result[i] = bestIdx;
            for (int d = outShape.length - 1; d >= 0; d--) {
                outIdx[d]++;
                if (outIdx[d] < outShape[d]) break;
                outIdx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Index of the minimum value along an axis. */
    public NDArray argmin(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis " + axis + " out of bounds for ndim " + shape.length);
        }
        int[] outShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != ax) outShape[j++] = shape[i];
        }
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];
        int[] outIdx = new int[outShape.length];
        for (int i = 0; i < outSize; i++) {
            int[] inIdx = new int[shape.length];
            for (int d = 0, j = 0; d < shape.length; d++) {
                if (d != ax) inIdx[d] = outIdx[j++];
            }
            inIdx[ax] = 0;
            float bestVal = get(inIdx);
            int bestIdx = 0;
            for (int j = 1; j < shape[ax]; j++) {
                inIdx[ax] = j;
                float val = get(inIdx);
                if (val < bestVal) {
                    bestVal = val;
                    bestIdx = j;
                }
            }
            result[i] = bestIdx;
            for (int d = outShape.length - 1; d >= 0; d--) {
                outIdx[d]++;
                if (outIdx[d] < outShape[d]) break;
                outIdx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Product of elements along an axis. */
    public NDArray prod(int axis) {
        return reduceAxis(axis, false, 1.0f, (a, b) -> a * b);
    }

    /** Variance along an axis (population variance, ddof=0). */
    public NDArray var(int axis, boolean keepDims) {
        NDArray m = mean(axis, true);
        NDArray diff = this.sub(m);
        NDArray sq = diff.square();
        return sq.mean(axis, keepDims);
    }

    /** Standard deviation along an axis. */
    public NDArray std(int axis, boolean keepDims) {
        return var(axis, keepDims).sqrt();
    }

    /** Counts non-zero elements. */
    public int countNonZero() {
        int count = 0;
        int n = size();
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            if (get(idx) != 0.0f) count++;
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return count;
    }

    // ================================================================
    // E10 — MatMul
    // ================================================================

    /** Vector dot product (1-D · 1-D → scalar wrapped in 0-D array). */
    public NDArray dot(NDArray other) {
        if (this.shape.length != 1 || other.shape.length != 1) {
            throw new IllegalArgumentException("dot requires both arrays to be 1-D");
        }
        if (this.shape[0] != other.shape[0]) {
            throw new IllegalArgumentException(
                "dot requires same length: " + this.shape[0] + " vs " + other.shape[0]);
        }
        float sum = 0;
        for (int i = 0; i < this.shape[0]; i++) {
            sum += this.get(i) * other.get(i);
        }
        return fromArray(new float[]{sum});
    }

    /**
     * Matrix multiplication.
     * <ul>
     *   <li>2-D × 2-D: (M,K) × (K,N) → (M,N)</li>
     *   <li>Batched: (...,M,K) × (...,K,N) → (...,M,N)</li>
     * </ul>
     */
    public NDArray matMul(NDArray other) {
        if (this.shape.length < 2 || other.shape.length < 2) {
            throw new IllegalArgumentException("matMul requires at least 2-D arrays");
        }
        int M = this.shape[this.shape.length - 2];
        int K = this.shape[this.shape.length - 1];
        int K2 = other.shape[other.shape.length - 2];
        int N = other.shape[other.shape.length - 1];
        if (K != K2) {
            throw new IllegalArgumentException(
                "matMul inner dimensions mismatch: " + K + " vs " + K2);
        }

        // Batch dimensions
        int[] batchA = java.util.Arrays.copyOf(this.shape, this.shape.length - 2);
        int[] batchB = java.util.Arrays.copyOf(other.shape, other.shape.length - 2);
        int[] batchShape = broadcastShapes(batchA, batchB);

        // Output shape = batchShape + [M, N]
        int[] outShape = new int[batchShape.length + 2];
        System.arraycopy(batchShape, 0, outShape, 0, batchShape.length);
        outShape[outShape.length - 2] = M;
        outShape[outShape.length - 1] = N;

        NDArray result = zeros(outShape);

        // Total batch size
        int batchSize = 1;
        for (int d : batchShape) batchSize *= d;

        int[] batchIdx = new int[batchShape.length];
        for (int b = 0; b < batchSize; b++) {
            // Map batchIdx → indices into A and B (with broadcast)
            int[] idxA = new int[this.shape.length];
            int[] idxB = new int[other.shape.length];
            for (int d = 0; d < batchShape.length; d++) {
                int dA = d - (batchShape.length - batchA.length);
                int dB = d - (batchShape.length - batchB.length);
                if (dA >= 0) idxA[dA] = (batchA[dA] == 1) ? 0 : batchIdx[d];
                if (dB >= 0) idxB[dB] = (batchB[dB] == 1) ? 0 : batchIdx[d];
            }

            int[] outIdx = new int[outShape.length];
            System.arraycopy(batchIdx, 0, outIdx, 0, batchShape.length);

            // 2D matmul for this batch position
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        idxA[idxA.length - 2] = i;
                        idxA[idxA.length - 1] = k;
                        idxB[idxB.length - 2] = k;
                        idxB[idxB.length - 1] = j;
                        sum += this.get(idxA) * other.get(idxB);
                    }
                    outIdx[outIdx.length - 2] = i;
                    outIdx[outIdx.length - 1] = j;
                    result.set(sum, outIdx);
                }
            }

            // Increment batch index
            for (int d = batchShape.length - 1; d >= 0; d--) {
                batchIdx[d]++;
                if (batchIdx[d] < batchShape[d]) break;
                batchIdx[d] = 0;
            }
        }
        return result;
    }

    // ================================================================
    // E11 — Slicing & Views
    // ================================================================

    /**
     * Returns a view into a sub-region of this array (zero-copy).
     *
     * @param ranges one {@link Slice} per axis
     */
    public NDArray slice(Slice... ranges) {
        if (ranges.length != shape.length) {
            throw new IllegalArgumentException(
                "expected " + shape.length + " slices but got " + ranges.length);
        }
        int newOffset = this.offset;
        int[] newShape = new int[shape.length];
        int[] newStrides = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            int start = ranges[i].start();
            int stop = Math.min(ranges[i].stop(), shape[i]);
            int step = ranges[i].step();
            if (step <= 0) {
                throw new IllegalArgumentException("slice step must be positive");
            }
            if (start < 0 || start >= shape[i] || stop < start) {
                throw new IllegalArgumentException(
                    "invalid slice [" + start + ":" + stop + "] for axis " + i + " with size " + shape[i]);
            }
            newOffset += start * strides[i];
            newShape[i] = (stop - start + step - 1) / step;
            newStrides[i] = strides[i] * step;
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = newOffset;
        return result;
    }

    /** Adds a dimension of size 1 at the given axis. */
    public NDArray expandDims(int axis) {
        int ndim = shape.length;
        if (axis < 0 || axis > ndim) {
            throw new IllegalArgumentException("axis " + axis + " out of range for ndim " + ndim);
        }
        int[] newShape = new int[ndim + 1];
        int[] newStrides = new int[ndim + 1];
        for (int i = 0; i < axis; i++) {
            newShape[i] = shape[i];
            newStrides[i] = strides[i];
        }
        newShape[axis] = 1;
        newStrides[axis] = 0;
        for (int i = axis; i < ndim; i++) {
            newShape[i + 1] = shape[i];
            newStrides[i + 1] = strides[i];
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = this.offset;
        return result;
    }

    /** Removes a dimension of size 1 at the given axis. */
    public NDArray squeeze(int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        if (shape[axis] != 1) {
            throw new IllegalArgumentException(
                "cannot squeeze axis " + axis + " with size " + shape[axis]);
        }
        int ndim = shape.length;
        int[] newShape = new int[ndim - 1];
        int[] newStrides = new int[ndim - 1];
        for (int i = 0, j = 0; i < ndim; i++) {
            if (i == axis) continue;
            newShape[j] = shape[i];
            newStrides[j] = strides[i];
            j++;
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = this.offset;
        return result;
    }

    /** Removes all dimensions of size 1. */
    public NDArray squeeze() {
        int count = 0;
        for (int s : shape) {
            if (s != 1) count++;
        }
        int[] newShape = new int[count];
        int[] newStrides = new int[count];
        int j = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                newShape[j] = shape[i];
                newStrides[j] = strides[i];
                j++;
            }
        }
        NDArray result = new NDArray();
        result.data = this.data;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = this.offset;
        return result;
    }

    // ================================================================
    // E12 — Creation & Random
    // ================================================================

    /** Creates a 1-D array: [start, start+step, ..., end). */
    public static NDArray arange(float start, float end, float step) {
        if (step == 0) throw new IllegalArgumentException("step must not be zero");
        int len = Math.max(0, (int) Math.ceil((end - start) / step));
        float[] data = new float[len];
        for (int i = 0; i < len; i++) {
            data[i] = start + i * step;
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = new int[]{len};
        arr.strides = new int[]{1};
        arr.offset = 0;
        return arr;
    }

    /** Creates a 1-D array of {@code num} evenly spaced values in [start, end]. */
    public static NDArray linspace(float start, float end, int num) {
        if (num < 1) throw new IllegalArgumentException("num must be >= 1");
        float[] data = new float[num];
        if (num == 1) {
            data[0] = start;
        } else {
            float step = (end - start) / (num - 1);
            for (int i = 0; i < num; i++) {
                data[i] = start + i * step;
            }
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = new int[]{num};
        arr.strides = new int[]{1};
        arr.offset = 0;
        return arr;
    }

    /** Creates an n×n identity matrix. */
    public static NDArray eye(int n) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0f;
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = new int[]{n, n};
        arr.strides = rowMajorStrides(arr.shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates a diagonal matrix from a 1-D vector. */
    public static NDArray diag(NDArray vector) {
        if (vector.shape.length != 1) {
            throw new IllegalArgumentException("diag requires a 1-D vector");
        }
        int n = vector.shape[0];
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = vector.data[vector.offset + i * vector.strides[0]];
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = new int[]{n, n};
        arr.strides = rowMajorStrides(arr.shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates an NDArray with standard normal random values N(0,1). */
    public static NDArray randn(int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        float[] data = new float[size];
        java.util.Random rng = globalRng;
        // Box-Muller transform
        for (int i = 0; i + 1 < size; i += 2) {
            double u1 = rng.nextDouble();
            double u2 = rng.nextDouble();
            while (u1 == 0) u1 = rng.nextDouble(); // avoid log(0)
            double r = Math.sqrt(-2 * Math.log(u1));
            double theta = 2 * Math.PI * u2;
            data[i] = (float) (r * Math.cos(theta));
            data[i + 1] = (float) (r * Math.sin(theta));
        }
        if (size % 2 != 0) {
            double u1 = rng.nextDouble();
            double u2 = rng.nextDouble();
            while (u1 == 0) u1 = rng.nextDouble();
            data[size - 1] = (float) (Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = shape.clone();
        arr.strides = rowMajorStrides(shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates an NDArray with uniform random values in [0, 1). */
    public static NDArray rand(int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        float[] data = new float[size];
        java.util.Random rng = globalRng;
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextFloat();
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = shape.clone();
        arr.strides = rowMajorStrides(shape);
        arr.offset = 0;
        return arr;
    }

    /** Creates an NDArray with uniform random values in [lo, hi). */
    public static NDArray uniform(float lo, float hi, int... shape) {
        int size = 1;
        for (int s : shape) size *= s;
        float[] data = new float[size];
        java.util.Random rng = globalRng;
        for (int i = 0; i < size; i++) {
            data[i] = lo + (hi - lo) * rng.nextFloat();
        }
        NDArray arr = new NDArray();
        arr.data = data;
        arr.shape = shape.clone();
        arr.strides = rowMajorStrides(shape);
        arr.offset = 0;
        return arr;
    }

    /** Shuffles an index array in-place (Fisher-Yates). */
    public static void shuffle(int[] indices) {
        java.util.Random rng = new java.util.Random();
        for (int i = indices.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
    }

    /** Fills all elements with the given value (in-place). */
    public void fill(float value) {
        int size = 1;
        for (int s : this.shape) size *= s;
        int ndim = this.shape.length;
        int[] idx = new int[ndim];
        for (int i = 0; i < size; i++) {
            int flatIdx = this.offset;
            for (int d = 0; d < ndim; d++) {
                flatIdx += idx[d] * this.strides[d];
            }
            this.data[flatIdx] = value;
            // increment idx
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < this.shape[d]) break;
                idx[d] = 0;
            }
        }
    }

    // ================================================================
    // E13 — Join & Transform
    // ================================================================

    /** Concatenates arrays along an existing axis. */
    public static NDArray concatenate(NDArray[] arrays, int axis) {
        if (arrays.length == 0) throw new IllegalArgumentException("need at least one array");
        int ndim = arrays[0].shape.length;
        int ax = axis < 0 ? ndim + axis : axis;
        if (ax < 0 || ax >= ndim) throw new IllegalArgumentException("axis out of range");
        // validate shapes on non-concat axes
        for (int k = 1; k < arrays.length; k++) {
            if (arrays[k].shape.length != ndim) {
                throw new IllegalArgumentException("all arrays must have the same number of dimensions");
            }
            for (int d = 0; d < ndim; d++) {
                if (d != ax && arrays[k].shape[d] != arrays[0].shape[d]) {
                    throw new IllegalArgumentException("shape mismatch on axis " + d);
                }
            }
        }
        // compute result shape
        int[] resultShape = arrays[0].shape.clone();
        int totalAxis = 0;
        for (NDArray a : arrays) totalAxis += a.shape[ax];
        resultShape[ax] = totalAxis;

        int size = 1;
        for (int s : resultShape) size *= s;
        float[] resultData = new float[size];
        int[] resultStrides = rowMajorStrides(resultShape);

        // copy data from each source array
        int axisOffset = 0;
        for (NDArray src : arrays) {
            int srcSize = 1;
            for (int s : src.shape) srcSize *= s;
            int srcNdim = src.shape.length;
            int[] idx = new int[srcNdim];
            for (int i = 0; i < srcSize; i++) {
                // read from src
                int srcFlat = src.offset;
                for (int d = 0; d < srcNdim; d++) srcFlat += idx[d] * src.strides[d];
                // write to result
                int dstFlat = 0;
                for (int d = 0; d < srcNdim; d++) {
                    int coord = (d == ax) ? idx[d] + axisOffset : idx[d];
                    dstFlat += coord * resultStrides[d];
                }
                resultData[dstFlat] = src.data[srcFlat];
                // increment idx
                for (int d = srcNdim - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < src.shape[d]) break;
                    idx[d] = 0;
                }
            }
            axisOffset += src.shape[ax];
        }

        NDArray result = new NDArray();
        result.data = resultData;
        result.shape = resultShape;
        result.strides = resultStrides;
        result.offset = 0;
        return result;
    }

    /** Stacks arrays along a new axis. */
    public static NDArray stack(NDArray[] arrays, int axis) {
        if (arrays.length == 0) throw new IllegalArgumentException("need at least one array");
        NDArray[] expanded = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            expanded[i] = arrays[i].expandDims(axis);
        }
        return concatenate(expanded, axis);
    }

    /**
     * Pads this array.
     *
     * @param padWidth {@code padWidth[i] = {before, after}} for axis i
     * @param value    the fill value for padded regions
     */
    public NDArray pad(int[][] padWidth, float value) {
        int ndim = this.shape.length;
        if (padWidth.length != ndim) {
            throw new IllegalArgumentException("padWidth length must match number of dimensions");
        }
        int[] newShape = new int[ndim];
        for (int d = 0; d < ndim; d++) {
            newShape[d] = padWidth[d][0] + this.shape[d] + padWidth[d][1];
        }
        int newSize = 1;
        for (int s : newShape) newSize *= s;
        float[] newData = new float[newSize];
        if (value != 0) {
            java.util.Arrays.fill(newData, value);
        }
        int[] newStrides = rowMajorStrides(newShape);

        // copy original data into padded array
        int srcSize = 1;
        for (int s : this.shape) srcSize *= s;
        int[] idx = new int[ndim];
        for (int i = 0; i < srcSize; i++) {
            int srcFlat = this.offset;
            int dstFlat = 0;
            for (int d = 0; d < ndim; d++) {
                srcFlat += idx[d] * this.strides[d];
                dstFlat += (idx[d] + padWidth[d][0]) * newStrides[d];
            }
            newData[dstFlat] = this.data[srcFlat];
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < this.shape[d]) break;
                idx[d] = 0;
            }
        }

        NDArray result = new NDArray();
        result.data = newData;
        result.shape = newShape;
        result.strides = newStrides;
        result.offset = 0;
        return result;
    }

    /** Reverses elements along the given axis. */
    public NDArray flip(int axis) {
        int ndim = this.shape.length;
        int ax = axis < 0 ? ndim + axis : axis;
        if (ax < 0 || ax >= ndim) throw new IllegalArgumentException("axis out of range");

        int size = 1;
        for (int s : this.shape) size *= s;
        float[] newData = new float[size];
        int[] newStrides = rowMajorStrides(this.shape);
        int[] idx = new int[ndim];
        for (int i = 0; i < size; i++) {
            // read from source with flipped axis
            int srcFlat = this.offset;
            int dstFlat = 0;
            for (int d = 0; d < ndim; d++) {
                int srcIdx = (d == ax) ? (this.shape[d] - 1 - idx[d]) : idx[d];
                srcFlat += srcIdx * this.strides[d];
                dstFlat += idx[d] * newStrides[d];
            }
            newData[dstFlat] = this.data[srcFlat];
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < this.shape[d]) break;
                idx[d] = 0;
            }
        }

        NDArray result = new NDArray();
        result.data = newData;
        result.shape = this.shape.clone();
        result.strides = newStrides;
        result.offset = 0;
        return result;
    }

    // ================================================================
    // E14 — Fancy Indexing
    // ================================================================

    /**
     * Selects rows/columns by index (embedding lookup).
     * <p>Example: {@code weight.indexSelect(0, new int[]{3, 0, 3, 7})}</p>
     */
    public NDArray indexSelect(int axis, int[] indices) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        int ndim = shape.length;
        int[] resultShape = shape.clone();
        resultShape[axis] = indices.length;

        int size = 1;
        for (int s : resultShape) size *= s;
        float[] resultData = new float[size];
        int[] resultStrides = rowMajorStrides(resultShape);

        int[] idx = new int[ndim];
        for (int i = 0; i < size; i++) {
            // compute source flat index: map idx[axis] -> indices[idx[axis]]
            int srcFlat = this.offset;
            for (int d = 0; d < ndim; d++) {
                int srcIdx = (d == axis) ? indices[idx[d]] : idx[d];
                srcFlat += srcIdx * this.strides[d];
            }
            resultData[i] = this.data[srcFlat];

            // increment N-dim index
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < resultShape[d]) break;
                idx[d] = 0;
            }
        }

        NDArray result = new NDArray();
        result.data = resultData;
        result.shape = resultShape;
        result.strides = resultStrides;
        result.offset = 0;
        return result;
    }

    /**
     * Scatter-adds {@code src} into this array (embedding backward).
     * <p>Equivalent to NumPy's {@code np.add.at(self, indices, src)}.</p>
     */
    public void scatterAdd(int axis, int[] indices, NDArray src) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        int ndim = src.shape.length;
        int srcSize = 1;
        for (int s : src.shape) srcSize *= s;

        int[] idx = new int[ndim];
        for (int i = 0; i < srcSize; i++) {
            // src flat index
            int srcFlat = src.offset;
            for (int d = 0; d < ndim; d++) {
                srcFlat += idx[d] * src.strides[d];
            }
            // self flat index: map idx[axis] -> indices[idx[axis]]
            int dstFlat = this.offset;
            for (int d = 0; d < ndim; d++) {
                int dstIdx = (d == axis) ? indices[idx[d]] : idx[d];
                dstFlat += dstIdx * this.strides[d];
            }
            this.data[dstFlat] += src.data[srcFlat];

            // increment N-dim index over src shape
            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < src.shape[d]) break;
                idx[d] = 0;
            }
        }
    }

    /**
     * Returns a new array where positions with {@code mask == 1.0} are
     * replaced by {@code value}.
     */
    public NDArray maskedFill(NDArray mask, float value) {
        if (!java.util.Arrays.equals(this.shape, mask.shape)) {
            throw new IllegalArgumentException("mask shape must match array shape");
        }
        int ndim = shape.length;
        int size = 1;
        for (int s : shape) size *= s;
        float[] resultData = new float[size];
        int[] resultStrides = rowMajorStrides(shape);

        int[] idx = new int[ndim];
        for (int i = 0; i < size; i++) {
            int selfFlat = this.offset;
            int maskFlat = mask.offset;
            for (int d = 0; d < ndim; d++) {
                selfFlat += idx[d] * this.strides[d];
                maskFlat += idx[d] * mask.strides[d];
            }
            resultData[i] = (mask.data[maskFlat] != 0.0f) ? value : this.data[selfFlat];

            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }

        NDArray result = new NDArray();
        result.data = resultData;
        result.shape = shape.clone();
        result.strides = resultStrides;
        result.offset = 0;
        return result;
    }

    /**
     * Element-wise conditional: picks from {@code x} where condition is
     * non-zero, else from {@code y}.
     */
    public static NDArray where(NDArray condition, NDArray x, NDArray y) {
        if (!java.util.Arrays.equals(condition.shape, x.shape)
                || !java.util.Arrays.equals(condition.shape, y.shape)) {
            throw new IllegalArgumentException("condition, x, and y must have the same shape");
        }
        int ndim = condition.shape.length;
        int size = 1;
        for (int s : condition.shape) size *= s;
        float[] resultData = new float[size];
        int[] resultStrides = rowMajorStrides(condition.shape);

        int[] idx = new int[ndim];
        for (int i = 0; i < size; i++) {
            int condFlat = condition.offset;
            int xFlat = x.offset;
            int yFlat = y.offset;
            for (int d = 0; d < ndim; d++) {
                condFlat += idx[d] * condition.strides[d];
                xFlat += idx[d] * x.strides[d];
                yFlat += idx[d] * y.strides[d];
            }
            resultData[i] = (condition.data[condFlat] != 0.0f) ? x.data[xFlat] : y.data[yFlat];

            for (int d = ndim - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < condition.shape[d]) break;
                idx[d] = 0;
            }
        }

        NDArray result = new NDArray();
        result.data = resultData;
        result.shape = condition.shape.clone();
        result.strides = resultStrides;
        result.offset = 0;
        return result;
    }

    // ================================================================
    // E15 — Capstone: Toolkit
    // ================================================================

    /** Lower-triangular matrix of size n (for causal masks). */
    public static NDArray tril(int n, int diagonal) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i * n + j] = (j <= i + diagonal) ? 1.0f : 0.0f;
            }
        }
        return fromArray(data, n, n);
    }

    /** Upper-triangular matrix of size n. */
    public static NDArray triu(int n, int diagonal) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i * n + j] = (j >= i + diagonal) ? 1.0f : 0.0f;
            }
        }
        return fromArray(data, n, n);
    }

    /** L2 norm along an axis. */
    public NDArray norm(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        // Build output shape (remove axis)
        int[] outShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != ax) outShape[j++] = shape[i];
        }
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];

        int[] outIdx = new int[outShape.length];
        for (int i = 0; i < outSize; i++) {
            int[] inIdx = new int[shape.length];
            for (int d = 0, j = 0; d < shape.length; d++) {
                if (d != ax) inIdx[d] = outIdx[j++];
            }
            float sumSq = 0;
            for (int j = 0; j < shape[ax]; j++) {
                inIdx[ax] = j;
                float v = get(inIdx);
                sumSq += v * v;
            }
            result[i] = (float) Math.sqrt(sumSq);

            for (int d = outShape.length - 1; d >= 0; d--) {
                outIdx[d]++;
                if (outIdx[d] < outShape[d]) break;
                outIdx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Differences between consecutive elements along an axis. */
    public NDArray diff(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        if (shape[ax] < 2) {
            throw new IllegalArgumentException("axis size must be >= 2 for diff");
        }
        // Result shape: same but axis dimension is shape[ax]-1
        int[] outShape = shape.clone();
        outShape[ax] = shape[ax] - 1;
        int outSize = 1;
        for (int s : outShape) outSize *= s;
        float[] result = new float[outSize];
        int[] outStrides = rowMajorStrides(outShape);

        int[] idx = new int[shape.length];
        for (int i = 0; i < outSize; i++) {
            // idx represents position in output; for source: next = idx[ax]+1
            int[] curIdx = idx.clone();
            int[] nextIdx = idx.clone();
            nextIdx[ax] = idx[ax] + 1;
            result[i] = get(nextIdx) - get(curIdx);

            for (int d = outShape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < outShape[d]) break;
                idx[d] = 0;
            }
        }
        return fromArray(result, outShape);
    }

    /** Computes the q-th percentile across all elements. */
    public NDArray percentile(float q) {
        if (q < 0 || q > 100) {
            throw new IllegalArgumentException("q must be between 0 and 100");
        }
        int n = size();
        float[] sorted = new float[n];
        // Flatten all elements
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            sorted[i] = get(idx);
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        java.util.Arrays.sort(sorted);
        float rank = q / 100.0f * (n - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        float frac = rank - lo;
        float result = sorted[lo] + frac * (sorted[hi] - sorted[lo]);
        return fromArray(new float[]{result}, 1);
    }

    /** Returns the indices that would sort along the given axis. */
    public NDArray argsort(int axis) {
        int ax = axis < 0 ? shape.length + axis : axis;
        if (ax < 0 || ax >= shape.length) {
            throw new IllegalArgumentException("axis out of range");
        }
        int outSize = size();
        float[] result = new float[outSize];
        int[] outStrides = rowMajorStrides(shape);

        // Build output shape (same as input)
        // For each "row" along axis, sort indices by value
        int[] outerShape = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != ax) outerShape[j++] = shape[i];
        }
        int outerSize = 1;
        for (int s : outerShape) outerSize *= s;

        int axLen = shape[ax];
        int[] outerIdx = new int[outerShape.length];
        for (int i = 0; i < outerSize; i++) {
            // Build base index
            int[] baseIdx = new int[shape.length];
            for (int d = 0, j = 0; d < shape.length; d++) {
                if (d != ax) baseIdx[d] = outerIdx[j++];
            }
            // Extract values along axis
            float[] vals = new float[axLen];
            Integer[] indices = new Integer[axLen];
            for (int j = 0; j < axLen; j++) {
                baseIdx[ax] = j;
                vals[j] = get(baseIdx);
                indices[j] = j;
            }
            // Sort indices by value (stable)
            final float[] v = vals;
            java.util.Arrays.sort(indices, (a, b) -> Float.compare(v[a], v[b]));
            // Write sorted indices to result
            for (int j = 0; j < axLen; j++) {
                baseIdx[ax] = j;
                int flat = 0;
                for (int d = 0; d < shape.length; d++) {
                    flat += baseIdx[d] * outStrides[d];
                }
                result[flat] = indices[j];
            }
            // Advance outer index
            for (int d = outerShape.length - 1; d >= 0; d--) {
                outerIdx[d]++;
                if (outerIdx[d] < outerShape[d]) break;
                outerIdx[d] = 0;
            }
        }
        return fromArray(result, shape.clone());
    }

    /** Returns sorted unique elements. */
    public NDArray unique() {
        int n = size();
        float[] all = new float[n];
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            all[i] = get(idx);
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        java.util.Arrays.sort(all);
        // Count unique
        int count = 1;
        for (int i = 1; i < n; i++) {
            if (all[i] != all[i - 1]) count++;
        }
        float[] result = new float[count];
        result[0] = all[0];
        int k = 1;
        for (int i = 1; i < n; i++) {
            if (all[i] != all[i - 1]) result[k++] = all[i];
        }
        return fromArray(result, count);
    }

    /**
     * Returns true if all elements are within {@code atol} of the
     * corresponding elements in {@code other}.
     */
    public boolean allClose(NDArray other, float atol) {
        if (!java.util.Arrays.equals(this.shape, other.shape)) {
            return false;
        }
        int n = size();
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            float a = this.get(idx);
            float b = other.get(idx);
            if (Math.abs(a - b) > atol) return false;
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return true;
    }

    /** Converts element type: float32 ↔ int8. */
    public NDArray astype(DType dtype) {
        int n = size();
        float[] result = new float[n];
        int[] idx = new int[shape.length];
        for (int i = 0; i < n; i++) {
            float v = get(idx);
            if (dtype == DType.INT8) {
                // Truncate to int, clamp to [-128, 127]
                int iv = (int) v;
                if (iv < -128) iv = -128;
                if (iv > 127) iv = 127;
                result[i] = iv;
            } else {
                // FLOAT32: just copy (already float)
                result[i] = v;
            }
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return fromArray(result, shape.clone());
    }
}
