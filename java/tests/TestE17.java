// TestE17.java — E17 Profiling & Compression test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Profiler;
import dev.tensorhero.tinytorch.Pruner;
import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.Layer;
import dev.tensorhero.tinytorch.Linear;
import dev.tensorhero.tinytorch.Sequential;
import dev.tensorhero.tinynum.NDArray;

public class TestE17 {
    public static void main(String[] args) {
        // ============================================================
        // Part 1: Profiler — countParams
        // ============================================================

        // 1. Single Linear(4, 3) with bias: 4*3 + 3 = 15 params
        Linear lin1 = new Linear(4, 3);
        int params1 = Profiler.countParams(lin1);
        emit("count_params_single_linear", String.valueOf(params1));

        // 2. Linear without bias: 4*3 = 12 params
        Linear lin2 = new Linear(4, 3, false);
        int params2 = Profiler.countParams(lin2);
        emit("count_params_no_bias", String.valueOf(params2));

        // 3. Sequential(Linear(4,8), Linear(8,2)): (4*8+8) + (8*2+2) = 58
        Sequential seq = new Sequential(new Linear(4, 8), new Linear(8, 2));
        int params3 = Profiler.countParams(seq);
        emit("count_params_sequential", String.valueOf(params3));

        // ============================================================
        // Part 1: Profiler — countFlops
        // ============================================================

        // 4. Linear(4,3) FLOPs: 2 * 4 * 3 = 24
        Linear lin3 = new Linear(4, 3);
        long flops1 = Profiler.countFlops(lin3, new int[]{1, 4});
        emit("count_flops_single_linear", String.valueOf(flops1));

        // 5. Sequential(Linear(128,64), Linear(64,10)):
        //    2*128*64 + 2*64*10 = 16384 + 1280 = 17664
        Sequential seq2 = new Sequential(new Linear(128, 64), new Linear(64, 10));
        long flops2 = Profiler.countFlops(seq2, new int[]{1, 128});
        emit("count_flops_sequential", String.valueOf(flops2));

        // 6. Empty model (no Linear layers): 0 FLOPs
        Sequential emptySeq = new Sequential();
        long flops3 = Profiler.countFlops(emptySeq, new int[]{1, 4});
        emit("count_flops_empty", String.valueOf(flops3));

        // ============================================================
        // Part 2: Pruner — magnitudePrune
        // ============================================================

        // 7. Prune with known values: [0.1, -0.8, 0.3, -0.05, 0.9, -0.2]
        //    abs = [0.1, 0.8, 0.3, 0.05, 0.9, 0.2]
        //    50% percentile of abs → threshold ≈ 0.25
        //    Values with |x| < 0.25: indices 0 (0.1), 3 (0.05), 5 (0.2)
        //    After prune: [0, -0.8, 0.3, 0, 0.9, 0]
        Tensor w1 = Tensor.fromArray(
            new float[]{0.1f, -0.8f, 0.3f, -0.05f, 0.9f, -0.2f}, 2, 3);
        Pruner.magnitudePrune(w1, 0.5f);
        // Count non-zeros: should be 3
        float nonZeros = 0;
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                if (Math.abs(w1.data.get(r, c)) > 1e-6f) nonZeros++;
            }
        }
        emit("prune_nonzero_count", String.valueOf((int) nonZeros));

        // 8. The large values should survive: -0.8 at (0,1) should be non-zero
        boolean bigSurvived = Math.abs(w1.data.get(0, 1) - (-0.8f)) < 1e-4f;
        emit("prune_big_value_survives", String.valueOf(bigSurvived).toLowerCase());

        // 9. The small values should be zero: 0.1 at (0,0) should be zero
        boolean smallPruned = Math.abs(w1.data.get(0, 0)) < 1e-6f;
        emit("prune_small_value_zero", String.valueOf(smallPruned).toLowerCase());

        // 10. Prune with sparsity=0 — nothing pruned
        Tensor w2 = Tensor.fromArray(
            new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
        Pruner.magnitudePrune(w2, 0.0f);
        float sum2 = w2.data.flatten().sum(0, true).get(0);
        boolean noPrune = Math.abs(sum2 - 10.0f) < 1e-4f;
        emit("prune_zero_sparsity", String.valueOf(noPrune).toLowerCase());

        // ============================================================
        // Part 2: Pruner — measureSparsity
        // ============================================================

        // 11. Model with no zeros: sparsity ≈ 0%
        Linear linA = new Linear(4, 3, false);
        // Override weight with all-ones to ensure no zeros
        linA.weight = Tensor.fromNDArray(NDArray.ones(new int[]{3, 4}));
        float sp1 = Pruner.measureSparsity(linA);
        boolean spZero = sp1 < 0.01f;
        emit("sparsity_all_ones", String.valueOf(spZero).toLowerCase());

        // 12. Model with known zeros: 4 out of 12 = 33.33%
        Linear linB = new Linear(4, 3, false);
        linB.weight = Tensor.fromArray(new float[]{
            1.0f, 0.0f, 2.0f, 0.0f,
            3.0f, 4.0f, 0.0f, 5.0f,
            6.0f, 7.0f, 8.0f, 0.0f
        }, 3, 4);
        float sp2 = Pruner.measureSparsity(linB);
        // 4/12 * 100 = 33.333...
        emit("sparsity_known_zeros", floatStr(sp2));

        // 13. Bias is NOT counted in sparsity (1D parameter)
        Linear linC = new Linear(2, 2);
        linC.weight = Tensor.fromNDArray(NDArray.ones(new int[]{2, 2}));
        linC.bias = Tensor.fromNDArray(NDArray.zeros(new int[]{2}));  // all-zero bias
        float sp3 = Pruner.measureSparsity(linC);
        // weight has 0 zeros out of 4 → 0%
        // bias is 1D → skipped
        boolean biasSkipped = sp3 < 0.01f;
        emit("sparsity_ignores_bias", String.valueOf(biasSkipped).toLowerCase());

        // 14. Prune then measure: roundtrip consistency
        Linear linD = new Linear(8, 4, false);
        // Set weight to known values: arange 0..31 reshaped (4, 8)
        //   has one zero at position (0,0)
        linD.weight = Tensor.fromNDArray(NDArray.arange(0, 32, 1).reshape(4, 8));
        Pruner.magnitudePrune(linD.weight, 0.3f);
        float sp4 = Pruner.measureSparsity(linD);
        // After pruning 30% of 32 values → ~10 zeros
        // Sparsity should be >= 25% (at least close to 30%)
        boolean spAfterPrune = sp4 >= 25.0f;
        emit("sparsity_after_prune", String.valueOf(spAfterPrune).toLowerCase());
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String shapeStr(int[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        return sb.toString();
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }
}
