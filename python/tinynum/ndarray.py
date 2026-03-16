"""N-dimensional array — the core data structure of tinynum.

Internally stores data in a flat list[float] with shape and strides
metadata, enabling zero-copy views for reshape, transpose, and slice.
"""

from __future__ import annotations
from typing import Sequence

from tinynum.dtype import DType
from tinynum.slice import Slice


class NDArray:
    """N-dimensional array with flat storage, shape, strides, and offset."""

    def __init__(self) -> None:
        self.data: list[float] = []       # flat storage
        self._shape: tuple[int, ...] = ()  # e.g. (2, 3, 4)
        self.strides: tuple[int, ...] = ()  # e.g. (12, 4, 1) row-major
        self.offset: int = 0              # for views/slices

    @staticmethod
    def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute row-major strides from shape."""
        if not shape:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    # ================================================================
    # E01 — Storage & Shape
    # ================================================================

    @staticmethod
    def from_array(data: list[float], *shape: int) -> "NDArray":
        """Creates an NDArray from a flat data list with the given shape.

        Raises:
            ValueError: if len(data) != product of shape
        """
        size = 1
        for s in shape:
            size *= s
        if len(data) != size:
            raise ValueError(
                f"data length {len(data)} does not match shape product {size}"
            )
        arr = NDArray()
        arr.data = list(data)
        arr._shape = shape
        arr.strides = NDArray._row_major_strides(shape)
        arr.offset = 0
        return arr

    @staticmethod
    def zeros(*shape: int) -> "NDArray":
        """Creates a zero-filled NDArray with the given shape."""
        return NDArray.full(0.0, *shape)

    @staticmethod
    def ones(*shape: int) -> "NDArray":
        """Creates a one-filled NDArray with the given shape."""
        return NDArray.full(1.0, *shape)

    @staticmethod
    def full(value: float, *shape: int) -> "NDArray":
        """Creates an NDArray filled with value."""
        size = 1
        for s in shape:
            size *= s
        arr = NDArray()
        arr.data = [float(value)] * size
        arr._shape = shape
        arr.strides = NDArray._row_major_strides(shape)
        arr.offset = 0
        return arr

    @staticmethod
    def zeros_like(other: "NDArray") -> "NDArray":
        """Creates a zero-filled NDArray with the same shape as other."""
        return NDArray.zeros(*other.get_shape())

    @staticmethod
    def ones_like(other: "NDArray") -> "NDArray":
        """Creates a one-filled NDArray with the same shape as other."""
        return NDArray.ones(*other.get_shape())

    def size(self) -> int:
        """Returns the total number of elements."""
        s = 1
        for d in self._shape:
            s *= d
        return s

    def ndim(self) -> int:
        """Returns the number of dimensions."""
        return len(self._shape)

    def get_shape(self) -> tuple[int, ...]:
        """Returns a copy of the shape tuple."""
        return tuple(self._shape)

    def __str__(self) -> str:
        """Pretty-prints the NDArray: e.g. '[[1.0, 2.0], [3.0, 4.0]]'."""
        return self._format_recursive(0, self.offset)

    def __repr__(self) -> str:
        return self.__str__()

    def _format_recursive(self, dim: int, flat_offset: int) -> str:
        if dim == len(self._shape) - 1:
            parts = []
            for i in range(self._shape[dim]):
                v = self.data[flat_offset + i * self.strides[dim]]
                parts.append(self._format_float(v))
            return "[" + ", ".join(parts) + "]"
        else:
            parts = []
            for i in range(self._shape[dim]):
                parts.append(
                    self._format_recursive(dim + 1, flat_offset + i * self.strides[dim])
                )
            return "[" + ", ".join(parts) + "]"

    @staticmethod
    def _format_float(v: float) -> str:
        # Ensure integer-valued floats show ".0"
        return str(float(v))

    # ================================================================
    # E02 — Strides & Indexing
    # ================================================================

    @staticmethod
    def compute_strides(shape: Sequence[int]) -> tuple[int, ...]:
        """Computes row-major strides for the given shape.

        Example: shape (3, 4, 5) → strides (20, 5, 1)
        """
        return NDArray._row_major_strides(tuple(shape))

    def get(self, *indices: int) -> float:
        """Gets the element at the given multi-dimensional indices.

        Uses: physical_index = offset + sum(index[i] * stride[i])
        """
        if len(indices) != len(self._shape):
            raise ValueError(
                f"expected {len(self._shape)} indices but got {len(indices)}"
            )
        idx = self.offset
        for i, ix in enumerate(indices):
            idx += ix * self.strides[i]
        return self.data[idx]

    def set(self, value: float, *indices: int) -> None:
        """Sets the element at the given multi-dimensional indices."""
        if len(indices) != len(self._shape):
            raise ValueError(
                f"expected {len(self._shape)} indices but got {len(indices)}"
            )
        idx = self.offset
        for i, ix in enumerate(indices):
            idx += ix * self.strides[i]
        self.data[idx] = float(value)

    def is_contiguous(self) -> bool:
        """Returns True if strides form a standard row-major contiguous layout."""
        return self.strides == NDArray._row_major_strides(self._shape)

    # ================================================================
    # E03 — Reshape
    # ================================================================

    def reshape(self, *new_shape: int) -> "NDArray":
        """Returns a view with a new shape (zero-copy when contiguous).

        Supports -1 for one dimension to auto-infer its size.
        """
        # Resolve -1
        neg_idx = -1
        known_product = 1
        for i, d in enumerate(new_shape):
            if d == -1:
                if neg_idx != -1:
                    raise ValueError("only one -1 allowed in reshape")
                neg_idx = i
            else:
                known_product *= d
        resolved = list(new_shape)
        sz = self.size()
        if neg_idx != -1:
            if sz % known_product != 0:
                raise ValueError(
                    f"cannot reshape array of size {sz} into shape with known product {known_product}"
                )
            resolved[neg_idx] = sz // known_product
        # Validate total size
        new_size = 1
        for d in resolved:
            new_size *= d
        if new_size != sz:
            raise ValueError(
                f"cannot reshape array of size {sz} into shape of size {new_size}"
            )
        resolved_tuple = tuple(resolved)
        # If non-contiguous, duplicate first
        src = self if self.is_contiguous() else self.duplicate()
        view = NDArray()
        view.data = src.data
        view._shape = resolved_tuple
        view.strides = NDArray._row_major_strides(resolved_tuple)
        view.offset = src.offset
        return view

    def flatten(self) -> "NDArray":
        """Flattens to a 1-D array. Equivalent to reshape(-1)."""
        return self.reshape(-1)

    def duplicate(self) -> "NDArray":
        """Returns a deep copy (always contiguous)."""
        new_data: list[float] = []
        idx = [0] * len(self._shape)
        for _ in range(self.size()):
            new_data.append(self.get(*idx))
            # Increment multi-dimensional index (row-major order)
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return NDArray.from_array(new_data, *self.get_shape())

    # ================================================================
    # E04 — Transpose
    # ================================================================

    def transpose(self, *axes: int) -> "NDArray":
        """Transpose the array.

        - No args: 2-D transpose (swap axis 0 and 1).
        - With args: N-D transpose, rearranges axes per the given permutation.

        Always zero-copy.
        """
        ndim = len(self._shape)
        if len(axes) == 0:
            if ndim != 2:
                raise ValueError("transpose() with no args requires a 2-D array")
            axes = (1, 0)
        if len(axes) != ndim:
            raise ValueError("axes length must match ndim")
        seen = set()
        for a in axes:
            if a < 0 or a >= ndim or a in seen:
                raise ValueError("invalid axes permutation")
            seen.add(a)
        new_shape = tuple(self._shape[axes[i]] for i in range(ndim))
        new_strides = tuple(self.strides[axes[i]] for i in range(ndim))
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = new_shape
        result.strides = new_strides
        result.offset = self.offset
        return result

    def swap_axes(self, axis1: int, axis2: int) -> "NDArray":
        """Swaps two axes (zero-copy)."""
        ndim = len(self._shape)
        if axis1 < 0 or axis1 >= ndim or axis2 < 0 or axis2 >= ndim:
            raise ValueError("axis out of range")
        axes = list(range(ndim))
        axes[axis1] = axis2
        axes[axis2] = axis1
        return self.transpose(*axes)

    # ================================================================
    # E05 — Unary Math
    # ================================================================

    def _apply_unary(self, fn) -> "NDArray":
        """Internal iterator: applies fn to every element, returns a new contiguous array."""
        import math as _math
        n = self.size()
        result: list[float] = []
        idx = [0] * len(self._shape)
        for _ in range(n):
            result.append(fn(self.get(*idx)))
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return NDArray.from_array(result, *self.get_shape())

    def neg(self) -> "NDArray":
        """Returns -x element-wise."""
        return self._apply_unary(lambda x: -x)

    def abs(self) -> "NDArray":
        """Returns |x| element-wise."""
        return self._apply_unary(lambda x: x if x >= 0 else -x)

    def exp(self) -> "NDArray":
        """Returns e^x element-wise."""
        import math as _math
        return self._apply_unary(_math.exp)

    def log(self) -> "NDArray":
        """Returns ln(x) element-wise."""
        import math as _math
        return self._apply_unary(_math.log)

    def sqrt(self) -> "NDArray":
        """Returns sqrt(x) element-wise."""
        import math as _math
        return self._apply_unary(_math.sqrt)

    def square(self) -> "NDArray":
        """Returns x² element-wise."""
        return self._apply_unary(lambda x: x * x)

    def tanh(self) -> "NDArray":
        """Returns tanh(x) element-wise."""
        import math as _math
        return self._apply_unary(_math.tanh)

    def sin(self) -> "NDArray":
        """Returns sin(x) element-wise."""
        import math as _math
        return self._apply_unary(_math.sin)

    def cos(self) -> "NDArray":
        """Returns cos(x) element-wise."""
        import math as _math
        return self._apply_unary(_math.cos)

    def sign(self) -> "NDArray":
        """Returns sgn(x) element-wise."""
        return self._apply_unary(lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)))

    def round(self) -> "NDArray":
        """Returns rounded values element-wise."""
        return self._apply_unary(lambda x: float(round(x)))

    def clip(self, min_val: float, max_val: float) -> "NDArray":
        """Clips values to [min_val, max_val] element-wise."""
        return self._apply_unary(lambda x: max(min_val, min(max_val, x)))

    def pow(self, p: float) -> "NDArray":
        """Returns x^p element-wise."""
        return self._apply_unary(lambda x: x ** p)

    # ================================================================
    # E06 — Binary Ops & Comparisons (same shape)
    # ================================================================

    def _apply_binary(self, other: "NDArray", fn) -> "NDArray":
        """Internal iterator: applies fn(a, b) element-wise with auto-broadcasting (E07)."""
        out_shape = NDArray.broadcast_shapes(self._shape, other._shape)
        a = self.broadcast_to(*out_shape)
        b = other.broadcast_to(*out_shape)
        n = 1
        for s in out_shape:
            n *= s
        result: list[float] = []
        idx = [0] * len(out_shape)
        for _ in range(n):
            result.append(fn(a.get(*idx), b.get(*idx)))
            for d in range(len(out_shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < out_shape[d]:
                    break
                idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    # --- Arithmetic (NDArray) ---

    def add(self, other: "NDArray") -> "NDArray":
        """Element-wise addition."""
        return self._apply_binary(other, lambda a, b: a + b)

    def sub(self, other: "NDArray") -> "NDArray":
        """Element-wise subtraction."""
        return self._apply_binary(other, lambda a, b: a - b)

    def mul(self, other: "NDArray") -> "NDArray":
        """Element-wise multiplication."""
        return self._apply_binary(other, lambda a, b: a * b)

    def div(self, other: "NDArray") -> "NDArray":
        """Element-wise division."""
        return self._apply_binary(other, lambda a, b: a / b)

    def pow_array(self, other: "NDArray") -> "NDArray":
        """Element-wise power: x^y."""
        return self._apply_binary(other, lambda a, b: a ** b)

    def maximum(self, other: "NDArray") -> "NDArray":
        """Element-wise maximum: max(x, y)."""
        return self._apply_binary(other, lambda a, b: a if a >= b else b)

    # --- Arithmetic (scalar) ---

    def add_scalar(self, scalar: float) -> "NDArray":
        """Adds a scalar to every element."""
        return self._apply_unary(lambda x: x + scalar)

    def sub_scalar(self, scalar: float) -> "NDArray":
        """Subtracts a scalar from every element."""
        return self._apply_unary(lambda x: x - scalar)

    def mul_scalar(self, scalar: float) -> "NDArray":
        """Multiplies every element by a scalar."""
        return self._apply_unary(lambda x: x * scalar)

    def div_scalar(self, scalar: float) -> "NDArray":
        """Divides every element by a scalar."""
        return self._apply_unary(lambda x: x / scalar)

    # --- Comparisons (NDArray) — returns 1.0 / 0.0 ---

    def eq(self, other: "NDArray") -> "NDArray":
        """Element-wise equal: returns 1.0 where x == y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a == b else 0.0)

    def neq(self, other: "NDArray") -> "NDArray":
        """Element-wise not-equal: returns 1.0 where x != y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a != b else 0.0)

    def gt(self, other: "NDArray") -> "NDArray":
        """Element-wise greater-than: returns 1.0 where x > y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a > b else 0.0)

    def gte(self, other: "NDArray") -> "NDArray":
        """Element-wise greater-than-or-equal: returns 1.0 where x >= y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a >= b else 0.0)

    def lt(self, other: "NDArray") -> "NDArray":
        """Element-wise less-than: returns 1.0 where x < y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a < b else 0.0)

    def lte(self, other: "NDArray") -> "NDArray":
        """Element-wise less-than-or-equal: returns 1.0 where x <= y."""
        return self._apply_binary(other, lambda a, b: 1.0 if a <= b else 0.0)

    # --- Comparisons (scalar) ---

    def eq_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x == scalar."""
        return self._apply_unary(lambda x: 1.0 if x == scalar else 0.0)

    def neq_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x != scalar."""
        return self._apply_unary(lambda x: 1.0 if x != scalar else 0.0)

    def gt_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x > scalar."""
        return self._apply_unary(lambda x: 1.0 if x > scalar else 0.0)

    def gte_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x >= scalar."""
        return self._apply_unary(lambda x: 1.0 if x >= scalar else 0.0)

    def lt_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x < scalar."""
        return self._apply_unary(lambda x: 1.0 if x < scalar else 0.0)

    def lte_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x <= scalar."""
        return self._apply_unary(lambda x: 1.0 if x <= scalar else 0.0)

    # ================================================================
    # E07 — Broadcasting
    # ================================================================

    @staticmethod
    def broadcast_shapes(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple[int, ...]:
        """Computes the broadcast-compatible output shape.

        Example: (3, 1) + (1, 4) → (3, 4)

        Raises:
            ValueError: if shapes are not broadcast-compatible
        """
        ndim = max(len(shape_a), len(shape_b))
        result = [0] * ndim
        for i in range(ndim - 1, -1, -1):
            a = shape_a[i - (ndim - len(shape_a))] if i - (ndim - len(shape_a)) >= 0 else 1
            b = shape_b[i - (ndim - len(shape_b))] if i - (ndim - len(shape_b)) >= 0 else 1
            if a == b:
                result[i] = a
            elif a == 1:
                result[i] = b
            elif b == 1:
                result[i] = a
            else:
                raise ValueError(
                    f"Shapes not broadcastable: {shape_a} vs {shape_b}"
                )
        return tuple(result)

    def broadcast_to(self, *target_shape: int) -> "NDArray":
        """Returns a view broadcast to the target shape (zero-copy, stride=0 trick)."""
        ndim = len(target_shape)
        if ndim < len(self._shape):
            raise ValueError("target ndim must be >= source ndim")
        new_strides = [0] * ndim
        for i in range(ndim - 1, -1, -1):
            src_idx = i - (ndim - len(self._shape))
            orig_dim = self._shape[src_idx] if src_idx >= 0 else 1
            if orig_dim == target_shape[i]:
                new_strides[i] = self.strides[src_idx] if src_idx >= 0 else 0
            elif orig_dim == 1:
                new_strides[i] = 0
            else:
                raise ValueError(
                    f"Cannot broadcast shape {self._shape} to {target_shape}"
                )
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = tuple(target_shape)
        result.strides = tuple(new_strides)
        result.offset = self.offset
        return result

    # ================================================================
    # E08 — Reduction: Sum & Mean
    # ================================================================

    def sum_all(self) -> float:
        """Returns the sum of all elements."""
        n = self.size()
        total = 0.0
        idx = [0] * len(self._shape)
        for _ in range(n):
            total += self.get(*idx)
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return total

    def mean_all(self) -> float:
        """Returns the mean of all elements."""
        return self.sum_all() / self.size()

    def sum(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Sum along a single axis."""
        ax = axis if axis >= 0 else len(self._shape) + axis
        if ax < 0 or ax >= len(self._shape):
            raise ValueError(f"axis {axis} out of bounds for ndim {len(self._shape)}")
        # Build output shape
        if keep_dims:
            out_shape = list(self._shape)
            out_shape[ax] = 1
            out_shape = tuple(out_shape)
        else:
            out_shape = tuple(s for i, s in enumerate(self._shape) if i != ax)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result: list[float] = []
        out_idx = [0] * len(out_shape)
        for _ in range(out_size):
            # Map out_idx → in_idx
            in_idx = [0] * len(self._shape)
            if keep_dims:
                for d in range(len(out_shape)):
                    in_idx[d] = out_idx[d]
            else:
                j = 0
                for d in range(len(self._shape)):
                    if d != ax:
                        in_idx[d] = out_idx[j]
                        j += 1
            total = 0.0
            for j in range(self._shape[ax]):
                in_idx[ax] = j
                total += self.get(*in_idx)
            result.append(total)
            # Advance out_idx
            for d in range(len(out_shape) - 1, -1, -1):
                out_idx[d] += 1
                if out_idx[d] < out_shape[d]:
                    break
                out_idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def mean(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Mean along a single axis."""
        ax = axis if axis >= 0 else len(self._shape) + axis
        s = self.sum(axis, keep_dims)
        count = self._shape[ax]
        return s.div_scalar(count)

    def sum_axes(self, axes: Sequence[int], keep_dims: bool = False) -> "NDArray":
        """Sum along multiple axes."""
        normalized = [a if a >= 0 else len(self._shape) + a for a in axes]
        sorted_axes = sorted(normalized, reverse=True)
        result = self
        for ax in sorted_axes:
            result = result.sum(ax, keep_dims)
        return result

    # ================================================================
    # E09 — Reduction: Max, Var & friends
    # ================================================================

    def _reduce_axis(self, axis: int, keep_dims: bool, init_val: float,
                     combine) -> "NDArray":
        """Generalized axis reduction."""
        ax = axis if axis >= 0 else len(self._shape) + axis
        if ax < 0 or ax >= len(self._shape):
            raise ValueError(f"axis {axis} out of bounds for ndim {len(self._shape)}")
        if keep_dims:
            out_shape = list(self._shape)
            out_shape[ax] = 1
            out_shape = tuple(out_shape)
        else:
            out_shape = tuple(s for i, s in enumerate(self._shape) if i != ax)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result: list[float] = []
        out_idx = [0] * len(out_shape)
        for _ in range(out_size):
            in_idx = [0] * len(self._shape)
            if keep_dims:
                for d in range(len(out_shape)):
                    in_idx[d] = out_idx[d]
            else:
                j = 0
                for d in range(len(self._shape)):
                    if d != ax:
                        in_idx[d] = out_idx[j]
                        j += 1
            in_idx[ax] = 0
            if init_val is None:
                acc = self.get(*in_idx)
                start = 1
            else:
                acc = init_val
                start = 0
            for j in range(start, self._shape[ax]):
                in_idx[ax] = j
                acc = combine(acc, self.get(*in_idx))
            result.append(acc)
            for d in range(len(out_shape) - 1, -1, -1):
                out_idx[d] += 1
                if out_idx[d] < out_shape[d]:
                    break
                out_idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def max(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Max along an axis."""
        return self._reduce_axis(axis, keep_dims, None,
                                 lambda a, b: a if a >= b else b)

    def min(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Min along an axis."""
        return self._reduce_axis(axis, keep_dims, None,
                                 lambda a, b: a if a <= b else b)

    def argmax(self, axis: int) -> "NDArray":
        """Index of the maximum value along an axis."""
        ax = axis if axis >= 0 else len(self._shape) + axis
        if ax < 0 or ax >= len(self._shape):
            raise ValueError(f"axis {axis} out of bounds for ndim {len(self._shape)}")
        out_shape = tuple(s for i, s in enumerate(self._shape) if i != ax)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result: list[float] = []
        out_idx = [0] * len(out_shape)
        for _ in range(out_size):
            in_idx = [0] * len(self._shape)
            j = 0
            for d in range(len(self._shape)):
                if d != ax:
                    in_idx[d] = out_idx[j]
                    j += 1
            in_idx[ax] = 0
            best_val = self.get(*in_idx)
            best_idx = 0
            for j in range(1, self._shape[ax]):
                in_idx[ax] = j
                val = self.get(*in_idx)
                if val > best_val:
                    best_val = val
                    best_idx = j
            result.append(float(best_idx))
            for d in range(len(out_shape) - 1, -1, -1):
                out_idx[d] += 1
                if out_idx[d] < out_shape[d]:
                    break
                out_idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def argmin(self, axis: int) -> "NDArray":
        """Index of the minimum value along an axis."""
        ax = axis if axis >= 0 else len(self._shape) + axis
        if ax < 0 or ax >= len(self._shape):
            raise ValueError(f"axis {axis} out of bounds for ndim {len(self._shape)}")
        out_shape = tuple(s for i, s in enumerate(self._shape) if i != ax)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result: list[float] = []
        out_idx = [0] * len(out_shape)
        for _ in range(out_size):
            in_idx = [0] * len(self._shape)
            j = 0
            for d in range(len(self._shape)):
                if d != ax:
                    in_idx[d] = out_idx[j]
                    j += 1
            in_idx[ax] = 0
            best_val = self.get(*in_idx)
            best_idx = 0
            for j in range(1, self._shape[ax]):
                in_idx[ax] = j
                val = self.get(*in_idx)
                if val < best_val:
                    best_val = val
                    best_idx = j
            result.append(float(best_idx))
            for d in range(len(out_shape) - 1, -1, -1):
                out_idx[d] += 1
                if out_idx[d] < out_shape[d]:
                    break
                out_idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def prod(self, axis: int) -> "NDArray":
        """Product of elements along an axis."""
        return self._reduce_axis(axis, False, 1.0,
                                 lambda a, b: a * b)

    def var(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Variance along an axis (population variance, ddof=0)."""
        m = self.mean(axis, keep_dims=True)
        diff = self.sub(m)
        sq = diff.square()
        return sq.mean(axis, keep_dims)

    def std(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Standard deviation along an axis."""
        return self.var(axis, keep_dims).sqrt()

    def count_nonzero(self) -> int:
        """Counts non-zero elements."""
        count = 0
        n = self.size()
        idx = [0] * len(self._shape)
        for _ in range(n):
            if self.get(*idx) != 0.0:
                count += 1
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return count

    # ================================================================
    # E10 — MatMul
    # ================================================================

    def dot(self, other: "NDArray") -> "NDArray":
        """Vector dot product (1-D · 1-D → scalar wrapped in 0-D array)."""
        if len(self._shape) != 1 or len(other._shape) != 1:
            raise ValueError("dot requires both arrays to be 1-D")
        if self._shape[0] != other._shape[0]:
            raise ValueError(
                f"dot requires same length: {self._shape[0]} vs {other._shape[0]}")
        s = 0.0
        for i in range(self._shape[0]):
            s += self.get(i) * other.get(i)
        return NDArray.from_array([s])

    def matmul(self, other: "NDArray") -> "NDArray":
        """Matrix multiplication.

        - 2-D × 2-D: (M,K) × (K,N) → (M,N)
        - Batched: (...,M,K) × (...,K,N) → (...,M,N)
        """
        if len(self._shape) < 2 or len(other._shape) < 2:
            raise ValueError("matmul requires at least 2-D arrays")
        M = self._shape[-2]
        K = self._shape[-1]
        K2 = other._shape[-2]
        N = other._shape[-1]
        if K != K2:
            raise ValueError(f"matmul inner dimensions mismatch: {K} vs {K2}")

        # Batch dimensions
        batch_a = self._shape[:-2]
        batch_b = other._shape[:-2]
        batch_shape = NDArray.broadcast_shapes(batch_a, batch_b)

        # Output shape = batch_shape + (M, N)
        out_shape = batch_shape + (M, N)
        result = NDArray.zeros(*out_shape)

        # Total batch size
        batch_size = 1
        for d in batch_shape:
            batch_size *= d

        batch_idx = [0] * len(batch_shape)
        for _ in range(batch_size):
            # Map batch_idx → indices into A and B (with broadcast)
            idx_a = [0] * len(self._shape)
            idx_b = [0] * len(other._shape)
            for d in range(len(batch_shape)):
                d_a = d - (len(batch_shape) - len(batch_a))
                d_b = d - (len(batch_shape) - len(batch_b))
                if d_a >= 0:
                    idx_a[d_a] = 0 if batch_a[d_a] == 1 else batch_idx[d]
                if d_b >= 0:
                    idx_b[d_b] = 0 if batch_b[d_b] == 1 else batch_idx[d]

            out_idx = list(batch_idx) + [0, 0]

            # 2D matmul for this batch position
            for i in range(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        idx_a[-2] = i
                        idx_a[-1] = k
                        idx_b[-2] = k
                        idx_b[-1] = j
                        s += self.get(*idx_a) * other.get(*idx_b)
                    out_idx[-2] = i
                    out_idx[-1] = j
                    result.set(s, *out_idx)

            # Increment batch index
            for d in range(len(batch_shape) - 1, -1, -1):
                batch_idx[d] += 1
                if batch_idx[d] < batch_shape[d]:
                    break
                batch_idx[d] = 0
        return result

    # ================================================================
    # E11 — Slicing & Views
    # ================================================================

    def slice(self, *ranges: Slice) -> "NDArray":
        """Returns a view into a sub-region of this array (zero-copy).

        Args:
            ranges: one Slice per axis
        """
        if len(ranges) != len(self._shape):
            raise ValueError(
                f"expected {len(self._shape)} slices but got {len(ranges)}")
        new_offset = self.offset
        new_shape = []
        new_strides = []
        for i in range(len(self._shape)):
            start = ranges[i].start
            stop = min(ranges[i].stop, self._shape[i])
            step = ranges[i].step
            if step <= 0:
                raise ValueError("slice step must be positive")
            if start < 0 or start >= self._shape[i] or stop < start:
                raise ValueError(
                    f"invalid slice [{start}:{stop}] for axis {i} with size {self._shape[i]}")
            new_offset += start * self.strides[i]
            new_shape.append((stop - start + step - 1) // step)
            new_strides.append(self.strides[i] * step)
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = tuple(new_shape)
        result.strides = tuple(new_strides)
        result.offset = new_offset
        return result

    def expand_dims(self, axis: int) -> "NDArray":
        """Adds a dimension of size 1 at the given axis."""
        ndim = len(self._shape)
        if axis < 0 or axis > ndim:
            raise ValueError(f"axis {axis} out of range for ndim {ndim}")
        new_shape = list(self._shape[:axis]) + [1] + list(self._shape[axis:])
        new_strides = list(self.strides[:axis]) + [0] + list(self.strides[axis:])
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = tuple(new_shape)
        result.strides = tuple(new_strides)
        result.offset = self.offset
        return result

    def squeeze_axis(self, axis: int) -> "NDArray":
        """Removes a dimension of size 1 at the given axis."""
        if axis < 0 or axis >= len(self._shape):
            raise ValueError("axis out of range")
        if self._shape[axis] != 1:
            raise ValueError(
                f"cannot squeeze axis {axis} with size {self._shape[axis]}")
        new_shape = list(self._shape[:axis]) + list(self._shape[axis + 1:])
        new_strides = list(self.strides[:axis]) + list(self.strides[axis + 1:])
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = tuple(new_shape)
        result.strides = tuple(new_strides)
        result.offset = self.offset
        return result

    def squeeze(self) -> "NDArray":
        """Removes all dimensions of size 1."""
        new_shape = []
        new_strides = []
        for i in range(len(self._shape)):
            if self._shape[i] != 1:
                new_shape.append(self._shape[i])
                new_strides.append(self.strides[i])
        result = NDArray.__new__(NDArray)
        result.data = self.data
        result._shape = tuple(new_shape)
        result.strides = tuple(new_strides)
        result.offset = self.offset
        return result

    # ================================================================
    # E12 — Creation & Random
    # ================================================================

    @staticmethod
    def arange(start: float, end: float, step: float) -> "NDArray":
        """Creates a 1-D array: [start, start+step, ..., end)."""
        if step == 0:
            raise ValueError("step must not be zero")
        import math
        length = max(0, math.ceil((end - start) / step))
        data = [start + i * step for i in range(length)]
        arr = NDArray()
        arr.data = data
        arr._shape = (length,)
        arr.strides = (1,)
        arr.offset = 0
        return arr

    @staticmethod
    def linspace(start: float, end: float, num: int) -> "NDArray":
        """Creates a 1-D array of num evenly spaced values in [start, end]."""
        if num < 1:
            raise ValueError("num must be >= 1")
        if num == 1:
            data = [float(start)]
        else:
            step = (end - start) / (num - 1)
            data = [start + i * step for i in range(num)]
        arr = NDArray()
        arr.data = data
        arr._shape = (num,)
        arr.strides = (1,)
        arr.offset = 0
        return arr

    @staticmethod
    def eye(n: int) -> "NDArray":
        """Creates an n×n identity matrix."""
        data = [0.0] * (n * n)
        for i in range(n):
            data[i * n + i] = 1.0
        arr = NDArray()
        arr.data = data
        arr._shape = (n, n)
        arr.strides = NDArray._row_major_strides((n, n))
        arr.offset = 0
        return arr

    @staticmethod
    def diag(vector: "NDArray") -> "NDArray":
        """Creates a diagonal matrix from a 1-D vector."""
        if len(vector._shape) != 1:
            raise ValueError("diag requires a 1-D vector")
        n = vector._shape[0]
        data = [0.0] * (n * n)
        for i in range(n):
            data[i * n + i] = vector.data[vector.offset + i * vector.strides[0]]
        arr = NDArray()
        arr.data = data
        arr._shape = (n, n)
        arr.strides = NDArray._row_major_strides((n, n))
        arr.offset = 0
        return arr

    @staticmethod
    def randn(*shape: int) -> "NDArray":
        """Creates an NDArray with standard normal random values N(0,1)."""
        import random
        import math
        size = 1
        for s in shape:
            size *= s
        data = [0.0] * size
        # Box-Muller transform
        i = 0
        while i + 1 < size:
            u1 = random.random()
            u2 = random.random()
            while u1 == 0:
                u1 = random.random()
            r = math.sqrt(-2 * math.log(u1))
            theta = 2 * math.pi * u2
            data[i] = r * math.cos(theta)
            data[i + 1] = r * math.sin(theta)
            i += 2
        if size % 2 != 0:
            u1 = random.random()
            u2 = random.random()
            while u1 == 0:
                u1 = random.random()
            data[size - 1] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        arr = NDArray()
        arr.data = data
        arr._shape = tuple(shape)
        arr.strides = NDArray._row_major_strides(tuple(shape))
        arr.offset = 0
        return arr

    @staticmethod
    def rand(*shape: int) -> "NDArray":
        """Creates an NDArray with uniform random values in [0, 1)."""
        import random
        size = 1
        for s in shape:
            size *= s
        data = [random.random() for _ in range(size)]
        arr = NDArray()
        arr.data = data
        arr._shape = tuple(shape)
        arr.strides = NDArray._row_major_strides(tuple(shape))
        arr.offset = 0
        return arr

    @staticmethod
    def uniform(lo: float, hi: float, *shape: int) -> "NDArray":
        """Creates an NDArray with uniform random values in [lo, hi)."""
        import random
        size = 1
        for s in shape:
            size *= s
        data = [lo + (hi - lo) * random.random() for _ in range(size)]
        arr = NDArray()
        arr.data = data
        arr._shape = tuple(shape)
        arr.strides = NDArray._row_major_strides(tuple(shape))
        arr.offset = 0
        return arr

    @staticmethod
    def shuffle(indices: list[int]) -> None:
        """Shuffles an index list in-place (Fisher-Yates)."""
        import random
        for i in range(len(indices) - 1, 0, -1):
            j = random.randint(0, i)
            indices[i], indices[j] = indices[j], indices[i]

    def fill(self, value: float) -> None:
        """Fills all elements with the given value (in-place)."""
        size = 1
        for s in self._shape:
            size *= s
        ndim = len(self._shape)
        idx = [0] * ndim
        for _ in range(size):
            flat_idx = self.offset
            for d in range(ndim):
                flat_idx += idx[d] * self.strides[d]
            self.data[flat_idx] = value
            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0

    # ================================================================
    # E13 — Join & Transform
    # ================================================================

    @staticmethod
    def concatenate(arrays: list["NDArray"], axis: int) -> "NDArray":
        """Concatenates arrays along an existing axis."""
        if len(arrays) == 0:
            raise ValueError("need at least one array")
        ndim = len(arrays[0]._shape)
        ax = axis if axis >= 0 else ndim + axis
        if ax < 0 or ax >= ndim:
            raise ValueError("axis out of range")
        for k in range(1, len(arrays)):
            if len(arrays[k]._shape) != ndim:
                raise ValueError("all arrays must have the same number of dimensions")
            for d in range(ndim):
                if d != ax and arrays[k]._shape[d] != arrays[0]._shape[d]:
                    raise ValueError(f"shape mismatch on axis {d}")
        # compute result shape
        result_shape = list(arrays[0]._shape)
        total_axis = sum(a._shape[ax] for a in arrays)
        result_shape[ax] = total_axis
        result_shape_t = tuple(result_shape)

        size = 1
        for s in result_shape:
            size *= s
        result_data = [0.0] * size
        result_strides = NDArray._row_major_strides(result_shape_t)

        axis_offset = 0
        for src in arrays:
            src_size = 1
            for s in src._shape:
                src_size *= s
            src_ndim = len(src._shape)
            idx = [0] * src_ndim
            for _ in range(src_size):
                src_flat = src.offset
                dst_flat = 0
                for d in range(src_ndim):
                    src_flat += idx[d] * src.strides[d]
                    coord = idx[d] + axis_offset if d == ax else idx[d]
                    dst_flat += coord * result_strides[d]
                result_data[dst_flat] = src.data[src_flat]
                for d in range(src_ndim - 1, -1, -1):
                    idx[d] += 1
                    if idx[d] < src._shape[d]:
                        break
                    idx[d] = 0
            axis_offset += src._shape[ax]

        arr = NDArray()
        arr.data = result_data
        arr._shape = result_shape_t
        arr.strides = result_strides
        arr.offset = 0
        return arr

    @staticmethod
    def stack(arrays: list["NDArray"], axis: int) -> "NDArray":
        """Stacks arrays along a new axis."""
        if len(arrays) == 0:
            raise ValueError("need at least one array")
        expanded = [a.expand_dims(axis) for a in arrays]
        return NDArray.concatenate(expanded, axis)

    def pad(self, pad_width: list[tuple[int, int]], value: float = 0.0) -> "NDArray":
        """Pads this array.

        Args:
            pad_width: [(before, after)] for each axis
            value: fill value for padded regions
        """
        ndim = len(self._shape)
        if len(pad_width) != ndim:
            raise ValueError("pad_width length must match number of dimensions")
        new_shape = tuple(pad_width[d][0] + self._shape[d] + pad_width[d][1] for d in range(ndim))
        new_size = 1
        for s in new_shape:
            new_size *= s
        new_data = [value] * new_size
        new_strides = NDArray._row_major_strides(new_shape)

        src_size = 1
        for s in self._shape:
            src_size *= s
        idx = [0] * ndim
        for _ in range(src_size):
            src_flat = self.offset
            dst_flat = 0
            for d in range(ndim):
                src_flat += idx[d] * self.strides[d]
                dst_flat += (idx[d] + pad_width[d][0]) * new_strides[d]
            new_data[dst_flat] = self.data[src_flat]
            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0

        arr = NDArray()
        arr.data = new_data
        arr._shape = new_shape
        arr.strides = new_strides
        arr.offset = 0
        return arr

    def flip(self, axis: int) -> "NDArray":
        """Reverses elements along the given axis."""
        ndim = len(self._shape)
        ax = axis if axis >= 0 else ndim + axis
        if ax < 0 or ax >= ndim:
            raise ValueError("axis out of range")

        size = 1
        for s in self._shape:
            size *= s
        new_strides = NDArray._row_major_strides(self._shape)
        new_data = [0.0] * size
        idx = [0] * ndim
        for _ in range(size):
            src_flat = self.offset
            dst_flat = 0
            for d in range(ndim):
                src_idx = (self._shape[d] - 1 - idx[d]) if d == ax else idx[d]
                src_flat += src_idx * self.strides[d]
                dst_flat += idx[d] * new_strides[d]
            new_data[dst_flat] = self.data[src_flat]
            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0

        arr = NDArray()
        arr.data = new_data
        arr._shape = self._shape
        arr.strides = new_strides
        arr.offset = 0
        return arr

    # ================================================================
    # E14 — Fancy Indexing
    # ================================================================

    def index_select(self, axis: int, indices: list[int]) -> "NDArray":
        """Selects rows/columns by index (embedding lookup).

        Example: weight.index_select(0, [3, 0, 3, 7])
        """
        if axis < 0 or axis >= len(self._shape):
            raise ValueError("axis out of range")
        ndim = len(self._shape)
        result_shape = list(self._shape)
        result_shape[axis] = len(indices)
        result_shape = tuple(result_shape)

        size = 1
        for s in result_shape:
            size *= s
        result_data = [0.0] * size
        result_strides = NDArray._row_major_strides(result_shape)

        idx = [0] * ndim
        for i in range(size):
            src_flat = self.offset
            for d in range(ndim):
                src_idx = indices[idx[d]] if d == axis else idx[d]
                src_flat += src_idx * self.strides[d]
            result_data[i] = self.data[src_flat]

            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < result_shape[d]:
                    break
                idx[d] = 0

        arr = NDArray()
        arr.data = result_data
        arr._shape = result_shape
        arr.strides = result_strides
        arr.offset = 0
        return arr

    def scatter_add(self, axis: int, indices: list[int], src: "NDArray") -> None:
        """Scatter-adds src into this array (embedding backward).

        Equivalent to numpy's np.add.at(self, indices, src).
        """
        if axis < 0 or axis >= len(self._shape):
            raise ValueError("axis out of range")
        ndim = len(src._shape)
        src_size = 1
        for s in src._shape:
            src_size *= s

        idx = [0] * ndim
        for i in range(src_size):
            src_flat = src.offset
            for d in range(ndim):
                src_flat += idx[d] * src.strides[d]

            dst_flat = self.offset
            for d in range(ndim):
                dst_idx = indices[idx[d]] if d == axis else idx[d]
                dst_flat += dst_idx * self.strides[d]
            self.data[dst_flat] += src.data[src_flat]

            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < src._shape[d]:
                    break
                idx[d] = 0

    def masked_fill(self, mask: "NDArray", value: float) -> "NDArray":
        """Returns a new array where positions with mask == 1.0 are replaced by value."""
        if self._shape != mask._shape:
            raise ValueError("mask shape must match array shape")
        ndim = len(self._shape)
        size = 1
        for s in self._shape:
            size *= s
        result_data = [0.0] * size
        result_strides = NDArray._row_major_strides(self._shape)

        idx = [0] * ndim
        for i in range(size):
            self_flat = self.offset
            mask_flat = mask.offset
            for d in range(ndim):
                self_flat += idx[d] * self.strides[d]
                mask_flat += idx[d] * mask.strides[d]
            result_data[i] = value if mask.data[mask_flat] != 0.0 else self.data[self_flat]

            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0

        arr = NDArray()
        arr.data = result_data
        arr._shape = self._shape
        arr.strides = result_strides
        arr.offset = 0
        return arr

    @staticmethod
    def where(condition: "NDArray", x: "NDArray", y: "NDArray") -> "NDArray":
        """Element-wise conditional: picks from x where condition is non-zero, else from y."""
        if condition._shape != x._shape or condition._shape != y._shape:
            raise ValueError("condition, x, and y must have the same shape")
        ndim = len(condition._shape)
        size = 1
        for s in condition._shape:
            size *= s
        result_data = [0.0] * size
        result_strides = NDArray._row_major_strides(condition._shape)

        idx = [0] * ndim
        for i in range(size):
            cond_flat = condition.offset
            x_flat = x.offset
            y_flat = y.offset
            for d in range(ndim):
                cond_flat += idx[d] * condition.strides[d]
                x_flat += idx[d] * x.strides[d]
                y_flat += idx[d] * y.strides[d]
            result_data[i] = x.data[x_flat] if condition.data[cond_flat] != 0.0 else y.data[y_flat]

            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < condition._shape[d]:
                    break
                idx[d] = 0

        arr = NDArray()
        arr.data = result_data
        arr._shape = condition._shape
        arr.strides = result_strides
        arr.offset = 0
        return arr

    # ================================================================
    # E15 — Capstone: Toolkit
    # ================================================================

    @staticmethod
    def tril(n: int, diagonal: int = 0) -> "NDArray":
        """Lower-triangular matrix of size n (for causal masks)."""
        data = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                data[i * n + j] = 1.0 if j <= i + diagonal else 0.0
        return NDArray.from_array(data, n, n)

    @staticmethod
    def triu(n: int, diagonal: int = 0) -> "NDArray":
        """Upper-triangular matrix of size n."""
        data = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                data[i * n + j] = 1.0 if j >= i + diagonal else 0.0
        return NDArray.from_array(data, n, n)

    def norm(self, axis: int) -> "NDArray":
        """L2 norm along an axis."""
        ndim = len(self._shape)
        ax = axis if axis >= 0 else ndim + axis
        if ax < 0 or ax >= ndim:
            raise ValueError("axis out of range")
        out_shape = []
        for i in range(ndim):
            if i != ax:
                out_shape.append(self._shape[i])
        out_shape = tuple(out_shape)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result = [0.0] * out_size

        outer_idx = [0] * len(out_shape)
        for i in range(out_size):
            in_idx = [0] * ndim
            j = 0
            for d in range(ndim):
                if d != ax:
                    in_idx[d] = outer_idx[j]
                    j += 1
            sum_sq = 0.0
            for k in range(self._shape[ax]):
                in_idx[ax] = k
                v = self.get(*in_idx)
                sum_sq += v * v
            result[i] = sum_sq ** 0.5

            for d in range(len(out_shape) - 1, -1, -1):
                outer_idx[d] += 1
                if outer_idx[d] < out_shape[d]:
                    break
                outer_idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def diff(self, axis: int) -> "NDArray":
        """Differences between consecutive elements along an axis."""
        ndim = len(self._shape)
        ax = axis if axis >= 0 else ndim + axis
        if ax < 0 or ax >= ndim:
            raise ValueError("axis out of range")
        if self._shape[ax] < 2:
            raise ValueError("axis size must be >= 2 for diff")
        out_shape = list(self._shape)
        out_shape[ax] = self._shape[ax] - 1
        out_shape = tuple(out_shape)
        out_size = 1
        for s in out_shape:
            out_size *= s
        result = [0.0] * out_size

        idx = [0] * ndim
        for i in range(out_size):
            cur_idx = list(idx)
            next_idx = list(idx)
            next_idx[ax] = idx[ax] + 1
            result[i] = self.get(*next_idx) - self.get(*cur_idx)

            for d in range(ndim - 1, -1, -1):
                idx[d] += 1
                if idx[d] < out_shape[d]:
                    break
                idx[d] = 0
        return NDArray.from_array(result, *out_shape)

    def percentile(self, q: float) -> "NDArray":
        """Computes the q-th percentile across all elements."""
        if q < 0 or q > 100:
            raise ValueError("q must be between 0 and 100")
        n = self.size()
        all_vals = [0.0] * n
        idx = [0] * len(self._shape)
        for i in range(n):
            all_vals[i] = self.get(*idx)
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        all_vals.sort()
        import math
        rank = q / 100.0 * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        frac = rank - lo
        val = all_vals[lo] + frac * (all_vals[hi] - all_vals[lo])
        return NDArray.from_array([val], 1)

    def argsort(self, axis: int) -> "NDArray":
        """Returns the indices that would sort along the given axis."""
        ndim = len(self._shape)
        ax = axis if axis >= 0 else ndim + axis
        if ax < 0 or ax >= ndim:
            raise ValueError("axis out of range")
        total = self.size()
        result = [0.0] * total
        result_strides = NDArray._row_major_strides(self._shape)

        outer_shape = []
        for i in range(ndim):
            if i != ax:
                outer_shape.append(self._shape[i])
        outer_size = 1
        for s in outer_shape:
            outer_size *= s

        ax_len = self._shape[ax]
        outer_idx = [0] * len(outer_shape)
        for i in range(outer_size):
            base_idx = [0] * ndim
            j = 0
            for d in range(ndim):
                if d != ax:
                    base_idx[d] = outer_idx[j]
                    j += 1
            vals = []
            for k in range(ax_len):
                base_idx[ax] = k
                vals.append(self.get(*base_idx))
            indices = list(range(ax_len))
            indices.sort(key=lambda x: vals[x])
            for k in range(ax_len):
                base_idx[ax] = k
                flat = 0
                for d in range(ndim):
                    flat += base_idx[d] * result_strides[d]
                result[flat] = float(indices[k])

            for d in range(len(outer_shape) - 1, -1, -1):
                outer_idx[d] += 1
                if outer_idx[d] < outer_shape[d]:
                    break
                outer_idx[d] = 0
        return NDArray.from_array(result, *self._shape)

    def unique(self) -> "NDArray":
        """Returns sorted unique elements."""
        n = self.size()
        all_vals = [0.0] * n
        idx = [0] * len(self._shape)
        for i in range(n):
            all_vals[i] = self.get(*idx)
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        all_vals.sort()
        uniq = [all_vals[0]]
        for i in range(1, n):
            if all_vals[i] != all_vals[i - 1]:
                uniq.append(all_vals[i])
        return NDArray.from_array(uniq, len(uniq))

    def all_close(self, other: "NDArray", atol: float = 1e-6) -> bool:
        """Returns True if all elements are within atol of other."""
        if self._shape != other._shape:
            return False
        n = self.size()
        idx = [0] * len(self._shape)
        for i in range(n):
            a = self.get(*idx)
            b = other.get(*idx)
            if abs(a - b) > atol:
                return False
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return True

    def astype(self, dtype: DType) -> "NDArray":
        """Converts element type: float32 ↔ int8."""
        n = self.size()
        result = [0.0] * n
        idx = [0] * len(self._shape)
        for i in range(n):
            v = self.get(*idx)
            if dtype == DType.INT8:
                iv = int(v)
                if iv < -128:
                    iv = -128
                if iv > 127:
                    iv = 127
                result[i] = float(iv)
            else:
                result[i] = v
            for d in range(len(self._shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
        return NDArray.from_array(result, *self._shape)
