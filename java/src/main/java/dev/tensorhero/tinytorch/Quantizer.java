package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * Quantizer — int8 quantization for model compression.
 *
 * <p>Maps float32 tensors to uint8 range [0, 255] with affine quantization,
 * and provides dequantization and quantized matrix multiplication.</p>
 */
public class Quantizer {

    /**
     * QuantizedTensor — holds quantized data with scale and zero-point metadata.
     */
    public static class QuantizedTensor {
        /** Quantized values (integers stored as float), range [0, 255]. */
        public NDArray data;
        /** Scale factor: maps one integer step to float range. */
        public float scale;
        /** Zero-point offset: the integer value that maps to float 0. */
        public float zeroPoint;
        /** Original tensor shape. */
        public int[] shape;

        public QuantizedTensor(NDArray data, float scale, float zeroPoint, int[] shape) {
            this.data = data;
            this.scale = scale;
            this.zeroPoint = zeroPoint;
            this.shape = shape;
        }
    }

    // ================================================================
    // E16 — Quantization & KV Cache
    // ================================================================

    /**
     * Quantizes a float32 NDArray to uint8 range [0, 255].
     *
     * @param tensor the float32 tensor to quantize
     * @return a QuantizedTensor with integer data, scale, and zero-point
     */
    public static QuantizedTensor quantize(NDArray tensor) {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Dequantizes back to float32.
     *
     * @param qt the quantized tensor
     * @return the recovered float32 NDArray
     */
    public static NDArray dequantize(QuantizedTensor qt) {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Quantized matrix multiplication (dequantize → float matMul).
     *
     * @param a first quantized tensor
     * @param b second quantized tensor
     * @return the matmul result as float32 NDArray
     */
    public static NDArray quantizedMatMul(QuantizedTensor a, QuantizedTensor b) {
        throw new UnsupportedOperationException("TODO: E16");
    }
}
