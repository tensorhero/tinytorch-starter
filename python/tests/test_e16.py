"""test_e16.py — E16 Quantization & KV Cache test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray
from tinytorch.quantizer import QuantizedTensor, quantize, dequantize, quantized_mat_mul
from tinytorch.kv_cache import KVCache


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in shape)


def main() -> None:
    # ============================================================
    # Part 1: Quantizer
    # ============================================================

    # Create test tensor: [0, 1, 2, ..., 11] reshaped to (3, 4)
    inp = NDArray.arange(0, 12, 1).reshape(3, 4)

    # 1. Quantize — shape preserved
    qt = quantize(inp)
    emit("quantize_shape", shape_str(qt.data.get_shape()))

    # 2. All quantized values >= 0
    q_min = qt.data.flatten().min(0, keep_dims=True).get(0)
    emit("quantize_range_min", str(q_min >= 0.0).lower())

    # 3. All quantized values <= 255
    q_max = qt.data.flatten().max(0, keep_dims=True).get(0)
    emit("quantize_range_max", str(q_max <= 255.0).lower())

    # 4. Dequantize — shape matches original
    recovered = dequantize(qt)
    emit("dequantize_shape", shape_str(recovered.get_shape()))

    # 5. Dequantize roundtrip — max error < 1% of range
    range_val = (
        inp.flatten().max(0, keep_dims=True).get(0)
        - inp.flatten().min(0, keep_dims=True).get(0)
    )
    max_error = inp.sub(recovered).abs().flatten().max(0, keep_dims=True).get(0)
    close_enough = max_error < range_val * 0.01
    emit("dequantize_close", str(close_enough).lower())

    # 6. Quantized matmul — output shape correct
    a = NDArray.arange(1, 7, 1).reshape(2, 3)  # (2, 3)
    b = NDArray.arange(1, 7, 1).reshape(3, 2)  # (3, 2)
    qa = quantize(a)
    qb = quantize(b)
    q_result = quantized_mat_mul(qa, qb)
    emit("quantized_matmul_shape", shape_str(q_result.get_shape()))

    # 7. Quantized matmul — error vs float matmul < 5%
    float_result = a.matmul(b)
    matmul_max_err = float_result.sub(q_result).abs().flatten().max(0, keep_dims=True).get(0)
    matmul_range = (
        float_result.flatten().max(0, keep_dims=True).get(0)
        - float_result.flatten().min(0, keep_dims=True).get(0)
    )
    matmul_close = matmul_range == 0.0 or matmul_max_err < matmul_range * 0.05
    emit("quantized_matmul_close", str(matmul_close).lower())

    # ============================================================
    # Part 2: KVCache
    # ============================================================

    # Use axis=1 (batch, seq, dim) — seq is the axis to grow
    cache = KVCache(1)

    # 8. Initial length = 0
    emit("kv_cache_initial_len", str(cache.current_len()))

    # First update: keys/values shape (2, 3, 4)
    keys1 = NDArray.arange(0, 24, 1).reshape(2, 3, 4)
    vals1 = NDArray.arange(24, 48, 1).reshape(2, 3, 4)
    cache.update(keys1, vals1)

    # 9. Length after first update = 3
    emit("kv_cache_update_len", str(cache.current_len()))

    # 10. Keys shape after first update = (2, 3, 4)
    emit("kv_cache_keys_shape", shape_str(cache.get_keys().get_shape()))

    # Second update: keys/values shape (2, 2, 4)
    keys2 = NDArray.arange(0, 16, 1).reshape(2, 2, 4)
    vals2 = NDArray.arange(16, 32, 1).reshape(2, 2, 4)
    cache.update(keys2, vals2)

    # 11. Length after two updates = 5
    emit("kv_cache_multi_update_len", str(cache.current_len()))

    # 12. Keys shape after two updates = (2, 5, 4)
    emit("kv_cache_multi_update_shape", shape_str(cache.get_keys().get_shape()))

    # 13. First element of cached values still correct
    #     vals1[0,0,0] = 24.0
    first_val = cache.get_values().get(0, 0, 0)
    emit("kv_cache_values_correct", str(abs(first_val - 24.0) < 0.001).lower())

    # 14. Reset clears the cache
    cache.reset()
    emit("kv_cache_reset_len", str(cache.current_len()))


if __name__ == "__main__":
    main()
