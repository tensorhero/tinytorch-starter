package dev.tensorhero.tinynum;

/**
 * Describes a slice range for one axis: [start, stop) with step.
 * Used by {@link NDArray#slice(Slice...)}.
 *
 * <p>Analogous to Python's {@code a[start:stop:step]}.</p>
 */
public record Slice(int start, int stop, int step) {

    /** Creates a slice [start, stop) with step 1. */
    public static Slice of(int start, int stop) {
        return new Slice(start, stop, 1);
    }

    /** Creates a slice [start, stop) with the given step. */
    public static Slice of(int start, int stop, int step) {
        return new Slice(start, stop, step);
    }

    /** Selects all elements along the axis (equivalent to {@code :} in Python). */
    public static Slice all() {
        return new Slice(0, Integer.MAX_VALUE, 1);
    }
}
