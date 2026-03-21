package dev.tensorhero.tinytorch;

import dev.tensorhero.tinynum.NDArray;
import dev.tensorhero.tinynum.Slice;

import java.util.Iterator;
import java.util.List;

/**
 * DataLoader — mini-batch iteration over a dataset.
 *
 * <p>Contains three nested types:</p>
 * <ul>
 *   <li>{@link Batch} — container for a (data, labels) pair of Tensors.</li>
 *   <li>{@link TensorDataset} — stores data and labels as 2D NDArrays.</li>
 *   <li>{@link DataLoader} — iterates over a TensorDataset in batches.</li>
 * </ul>
 */
public class DataLoader implements Iterable<DataLoader.Batch> {

    // ================================================================
    // Batch — simple container
    // ================================================================

    /**
     * A single batch of data and labels.
     */
    public static class Batch {
        /** Input data tensor. */
        public Tensor data;
        /** Label tensor. */
        public Tensor labels;

        public Batch(Tensor data, Tensor labels) {
            this.data = data;
            this.labels = labels;
        }
    }

    // ================================================================
    // TensorDataset — holds data + labels NDArrays
    // ================================================================

    /**
     * A dataset backed by two 2D NDArrays (data and labels).
     *
     * <p>The first dimension of both arrays must match (number of samples).
     * {@code get(i)} returns a {@link Batch} containing the i-th row of each array.</p>
     */
    public static class TensorDataset {

        /** Raw data array, shape [N, inputDim]. */
        public NDArray data;

        /** Raw labels array, shape [N, outputDim]. */
        public NDArray labels;

        // ================================================================
        // E10 — DataLoader & MLP
        // ================================================================

        /**
         * Creates a TensorDataset.
         *
         * @param data   2D NDArray of shape [N, inputDim]
         * @param labels 2D NDArray of shape [N, outputDim]
         * @throws IllegalArgumentException if first dimensions don't match
         */
        public TensorDataset(NDArray data, NDArray labels) {
            throw new UnsupportedOperationException("TODO: E10");
        }

        /**
         * Returns the number of samples.
         */
        public int size() {
            throw new UnsupportedOperationException("TODO: E10");
        }

        /**
         * Returns the i-th sample as a Batch of 1D Tensors.
         *
         * @param index sample index in [0, size())
         * @return Batch with data shape [inputDim] and labels shape [outputDim]
         */
        public Batch get(int index) {
            throw new UnsupportedOperationException("TODO: E10");
        }
    }

    // ================================================================
    // DataLoader — batched iteration
    // ================================================================

    private TensorDataset dataset;
    private int batchSize;
    private boolean shuffle;

    /**
     * Creates a DataLoader.
     *
     * @param dataset   the dataset to iterate over
     * @param batchSize number of samples per batch
     * @param shuffle   whether to shuffle indices each iteration
     */
    public DataLoader(TensorDataset dataset, int batchSize, boolean shuffle) {
        throw new UnsupportedOperationException("TODO: E10");
    }

    /**
     * Returns the number of batches (ceil division).
     */
    public int numBatches() {
        throw new UnsupportedOperationException("TODO: E10");
    }

    /**
     * Returns an iterator over batches.
     * If shuffle is true, indices are randomly permuted each time.
     */
    @Override
    public Iterator<Batch> iterator() {
        throw new UnsupportedOperationException("TODO: E10");
    }
}
