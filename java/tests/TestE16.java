// TestE16.java — E16 Quantization & KV Cache test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Quantizer;
import dev.tensorhero.tinytorch.Quantizer.QuantizedTensor;
import dev.tensorhero.tinytorch.KVCache;
import dev.tensorhero.tinynum.NDArray;

public class TestE16 {
    public static void main(String[] args) {
        // ============================================================
        // Part 1: Quantizer
        // ============================================================

        // Create test tensor: [0, 1, 2, ..., 11] reshaped to (3, 4)
        NDArray input = NDArray.arange(0, 12, 1).reshape(3, 4);

        // 1. Quantize — shape preserved
        QuantizedTensor qt = Quantizer.quantize(input);
        emit("quantize_shape", shapeStr(qt.data.shape()));

        // 2. All quantized values >= 0
        float qMin = qt.data.flatten().min(0, true).get(0);
        emit("quantize_range_min", String.valueOf(qMin >= 0.0f).toLowerCase());

        // 3. All quantized values <= 255
        float qMax = qt.data.flatten().max(0, true).get(0);
        emit("quantize_range_max", String.valueOf(qMax <= 255.0f).toLowerCase());

        // 4. Dequantize — shape matches original
        NDArray recovered = Quantizer.dequantize(qt);
        emit("dequantize_shape", shapeStr(recovered.shape()));

        // 5. Dequantize roundtrip — max error < 1% of range
        float range = input.flatten().max(0, true).get(0)
                    - input.flatten().min(0, true).get(0);
        float maxError = input.sub(recovered).abs().flatten().max(0, true).get(0);
        boolean closeEnough = maxError < range * 0.01f;
        emit("dequantize_close", String.valueOf(closeEnough).toLowerCase());

        // 6. Quantized matmul — output shape correct
        NDArray a = NDArray.arange(1, 7, 1).reshape(2, 3);   // (2, 3)
        NDArray b = NDArray.arange(1, 7, 1).reshape(3, 2);   // (3, 2)
        QuantizedTensor qa = Quantizer.quantize(a);
        QuantizedTensor qb = Quantizer.quantize(b);
        NDArray qResult = Quantizer.quantizedMatMul(qa, qb);
        emit("quantized_matmul_shape", shapeStr(qResult.shape()));

        // 7. Quantized matmul — error vs float matmul < 5%
        NDArray floatResult = a.matMul(b);
        float matmulMaxErr = floatResult.sub(qResult).abs().flatten().max(0, true).get(0);
        float matmulRange = floatResult.flatten().max(0, true).get(0)
                          - floatResult.flatten().min(0, true).get(0);
        boolean matmulClose = matmulRange == 0.0f || matmulMaxErr < matmulRange * 0.05f;
        emit("quantized_matmul_close", String.valueOf(matmulClose).toLowerCase());

        // ============================================================
        // Part 2: KVCache
        // ============================================================

        // Use axis=1 (batch, seq, dim) — seq is the axis to grow
        KVCache cache = new KVCache(1);

        // 8. Initial length = 0
        emit("kv_cache_initial_len", String.valueOf(cache.currentLen()));

        // First update: keys/values shape (2, 3, 4)
        NDArray keys1 = NDArray.arange(0, 24, 1).reshape(2, 3, 4);
        NDArray vals1 = NDArray.arange(24, 48, 1).reshape(2, 3, 4);
        cache.update(keys1, vals1);

        // 9. Length after first update = 3
        emit("kv_cache_update_len", String.valueOf(cache.currentLen()));

        // 10. Keys shape after first update = (2, 3, 4)
        emit("kv_cache_keys_shape", shapeStr(cache.getKeys().shape()));

        // Second update: keys/values shape (2, 2, 4)
        NDArray keys2 = NDArray.arange(0, 16, 1).reshape(2, 2, 4);
        NDArray vals2 = NDArray.arange(16, 32, 1).reshape(2, 2, 4);
        cache.update(keys2, vals2);

        // 11. Length after two updates = 5
        emit("kv_cache_multi_update_len", String.valueOf(cache.currentLen()));

        // 12. Keys shape after two updates = (2, 5, 4)
        emit("kv_cache_multi_update_shape", shapeStr(cache.getKeys().shape()));

        // 13. First element of cached values still correct
        //     vals1[0,0,0] = 24.0
        float firstVal = cache.getValues().get(0, 0, 0);
        emit("kv_cache_values_correct", String.valueOf(Math.abs(firstVal - 24.0f) < 0.001f).toLowerCase());

        // 14. Reset clears the cache
        cache.reset();
        emit("kv_cache_reset_len", String.valueOf(cache.currentLen()));
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
