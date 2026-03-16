"""Quantizer — int8 affine quantization for NDArray tensors.

Implements quantize (float32 -> uint8 range), dequantize (uint8 -> float32),
and quantized matrix multiplication.
"""

from __future__ import annotations

from tinynum import NDArray


class QuantizedTensor:
    """Holds quantized data along with scale and zero-point metadata."""

    # ================================================================
    # E16 — Quantization & KV Cache
    # ================================================================

    def __init__(self, data: NDArray, scale: float, zero_point: float, shape: list[int]) -> None:
        raise NotImplementedError("TODO: E16")


def quantize(tensor: NDArray) -> QuantizedTensor:
    """Quantize a float32 NDArray to uint8 range [0, 255]."""
    raise NotImplementedError("TODO: E16")


def dequantize(qt: QuantizedTensor) -> NDArray:
    """Dequantize a QuantizedTensor back to float32."""
    raise NotImplementedError("TODO: E16")


def quantized_mat_mul(a: QuantizedTensor, b: QuantizedTensor) -> NDArray:
    """Approximate matrix multiplication via dequantize -> matmul."""
    raise NotImplementedError("TODO: E16")
