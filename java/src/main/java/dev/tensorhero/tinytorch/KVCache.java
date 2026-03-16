package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;

/**
 * KVCache — caches Key and Value tensors for efficient autoregressive generation.
 *
 * <p>Stores and concatenates Key/Value arrays along a specified axis,
 * avoiding redundant computation during Transformer inference.</p>
 */
public class KVCache {

    /** Accumulated keys, or null if empty. */
    public NDArray keys;

    /** Accumulated values, or null if empty. */
    public NDArray values;

    /** Concatenation axis. */
    public int axis;

    /** Current cached sequence length. */
    public int currentLen;

    // ================================================================
    // E16 — Quantization & KV Cache
    // ================================================================

    /**
     * Creates a KVCache that concatenates along the given axis.
     *
     * @param axis the axis along which to concatenate K/V
     */
    public KVCache(int axis) {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Appends new keys and values to the cache.
     *
     * @param newKeys   new key tensor to append
     * @param newValues new value tensor to append
     */
    public void update(NDArray newKeys, NDArray newValues) {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Returns the accumulated keys.
     */
    public NDArray getKeys() {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Returns the accumulated values.
     */
    public NDArray getValues() {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Returns the current cached sequence length.
     */
    public int currentLen() {
        throw new UnsupportedOperationException("TODO: E16");
    }

    /**
     * Resets the cache (clears all stored keys and values).
     */
    public void reset() {
        throw new UnsupportedOperationException("TODO: E16");
    }
}
