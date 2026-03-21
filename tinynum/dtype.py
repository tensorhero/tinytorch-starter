"""Supported data types for NDArray.

The course uses FLOAT32 throughout. INT8 is introduced in E15
to prepare for int8 quantization in tinytorch.
"""

from enum import Enum


class DType(Enum):
    FLOAT32 = "float32"
    INT8 = "int8"
