"""Tensor — the core data structure of TinyTorch.

Wraps an NDArray from TinyNum and adds gradient metadata
(requires_grad, grad, grad_fn) for automatic differentiation.
In E01-E04 these fields remain dormant.
"""

from __future__ import annotations

from typing import Optional

from tinynum import NDArray
from tinytorch.function import Function


class Tensor:
    """Tensor with autograd support, backed by TinyNum NDArray."""

    def __init__(self) -> None:
        self.data: Optional[NDArray] = None        # the underlying NDArray
        self.requires_grad: bool = False            # dormant until E05
        self.grad: Optional["Tensor"] = None        # dormant until E05
        self.grad_fn: Optional[Function] = None     # dormant until E05

    # ================================================================
    # E01 — Tensor Class
    # ================================================================

    # --- Factory methods ---

    @staticmethod
    def from_ndarray(data: NDArray) -> "Tensor":
        """Creates a Tensor wrapping an existing NDArray.

        Args:
            data: the NDArray to wrap

        Returns:
            a new Tensor
        """
        raise NotImplementedError("TODO: E01")

    @staticmethod
    def from_array(data: list[float], *shape: int) -> "Tensor":
        """Creates a Tensor from a flat data list with the given shape.

        Args:
            data: the flat data list
            *shape: the desired shape

        Returns:
            a new Tensor
        """
        raise NotImplementedError("TODO: E01")

    @staticmethod
    def zeros(*shape: int) -> "Tensor":
        """Creates a zero-filled Tensor with the given shape."""
        raise NotImplementedError("TODO: E01")

    @staticmethod
    def ones(*shape: int) -> "Tensor":
        """Creates a one-filled Tensor with the given shape."""
        raise NotImplementedError("TODO: E01")

    @staticmethod
    def full(value: float, *shape: int) -> "Tensor":
        """Creates a Tensor filled with the given value."""
        raise NotImplementedError("TODO: E01")

    @staticmethod
    def randn(*shape: int) -> "Tensor":
        """Creates a Tensor with standard normal random values (mean=0, std=1)."""
        raise NotImplementedError("TODO: E01")

    # --- Element-wise operations (delegate to NDArray) ---

    def add(self, other: "Tensor") -> "Tensor":
        """Element-wise addition."""
        raise NotImplementedError("TODO: E01")

    def sub(self, other: "Tensor") -> "Tensor":
        """Element-wise subtraction."""
        raise NotImplementedError("TODO: E01")

    def mul(self, other: "Tensor") -> "Tensor":
        """Element-wise multiplication."""
        raise NotImplementedError("TODO: E01")

    def div(self, other: "Tensor") -> "Tensor":
        """Element-wise division."""
        raise NotImplementedError("TODO: E01")

    def mat_mul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        raise NotImplementedError("TODO: E01")

    # --- Reduction operations ---

    def sum(self, axis: int, keep_dims: bool = False) -> "Tensor":
        """Sum along the given axis.

        Args:
            axis: the axis to sum along
            keep_dims: whether to keep the reduced dimension
        """
        raise NotImplementedError("TODO: E01")

    def mean(self, axis: int, keep_dims: bool = False) -> "Tensor":
        """Mean along the given axis.

        Args:
            axis: the axis to average along
            keep_dims: whether to keep the reduced dimension
        """
        raise NotImplementedError("TODO: E01")

    # --- Shape operations ---

    def reshape(self, *shape: int) -> "Tensor":
        """Returns a new Tensor with the given shape."""
        raise NotImplementedError("TODO: E01")

    def transpose(self, *axes: int) -> "Tensor":
        """Returns a transposed Tensor with axes permuted."""
        raise NotImplementedError("TODO: E01")

    # --- Properties ---

    def shape(self) -> tuple[int, ...]:
        """Returns the shape of this tensor."""
        raise NotImplementedError("TODO: E01")

    def ndim(self) -> int:
        """Returns the number of dimensions."""
        raise NotImplementedError("TODO: E01")

    def size(self) -> int:
        """Returns the total number of elements."""
        raise NotImplementedError("TODO: E01")

    def __str__(self) -> str:
        """Returns a string representation (delegates to NDArray)."""
        raise NotImplementedError("TODO: E01")

    def __repr__(self) -> str:
        return f"Tensor({self.__str__()})"

    # ================================================================
    # E05 — Computation Graph
    # ================================================================
    # When requires_grad is True on any input, add(), sub(), mul(), div()
    # should create the corresponding Backward function, call forward(),
    # and set result.requires_grad = True and result.grad_fn = fn.
    # Otherwise, use the fast path (direct NDArray delegation from E01).

    # ================================================================
    # E06 — More Backward Ops
    # ================================================================
    # Extend graph recording to: mat_mul, sum, mean, reshape, transpose.
    # Add new methods: exp(), log() (with ExpBackward, LogBackward).
    # Modify activations: relu, sigmoid, tanh, gelu → use corresponding Backward.
    # Modify losses: cross_entropy → CrossEntropyBackward; mse → use Tensor ops.
    # Modify Dropout: forward → DropoutBackward.

    def exp(self) -> "Tensor":
        """Element-wise exponential: y = exp(x)."""
        raise NotImplementedError("TODO: E06")

    def log(self) -> "Tensor":
        """Element-wise natural log: y = log(x)."""
        raise NotImplementedError("TODO: E06")

    # ================================================================
    # E07 — Backpropagation
    # ================================================================
    # Implement backward(), topological_sort(), reduce_broadcast_grad(), and no_grad().
    # Also add _grad_enabled check to all graph-recording methods (add, sub, mul, etc.).

    def backward(self) -> None:
        """Backpropagation: compute gradients for all tensors in the computation graph.

        Must be called on a scalar tensor (size == 1).
        Sets .grad on each tensor with requires_grad == True.
        """
        raise NotImplementedError("TODO: E07")

    @staticmethod
    def no_grad(fn):
        """Execute a callable with gradient computation disabled.

        Args:
            fn: a zero-argument callable to execute

        Returns:
            the result of fn()
        """
        raise NotImplementedError("TODO: E07")


# Module-level flag: global switch to enable/disable gradient computation.
_grad_enabled = True


def topological_sort(root: "Tensor") -> list["Tensor"]:
    """Topological sort of the computation graph rooted at root.

    Uses DFS post-order: leaves first, root last.

    Args:
        root: the root tensor (typically the loss)

    Returns:
        list in topological order (leaves -> root)
    """
    raise NotImplementedError("TODO: E07")


def reduce_broadcast_grad(grad, target_shape: tuple[int, ...]):
    """Reduce a gradient that was broadcast during forward pass back to the target shape.

    Args:
        grad: the NDArray gradient to reduce
        target_shape: the original shape to reduce to

    Returns:
        the reduced NDArray gradient matching target_shape
    """
    raise NotImplementedError("TODO: E07")
