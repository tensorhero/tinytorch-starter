"""Autograd function — records forward computation and computes backward gradients.

This is a placeholder for E01-E04. Full implementation starts in E05.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinynum import NDArray
    from tinytorch.tensor import Tensor


class Function:
    """Base class for autograd functions (E05+)."""

    # ================================================================
    # E05 — Computation Graph (placeholder)
    # ================================================================

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        """Forward pass: compute output tensors from inputs."""
        raise NotImplementedError("TODO: E05")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        """Backward pass: compute gradients of inputs given gradient of output."""
        raise NotImplementedError("TODO: E05")

    def get_inputs(self) -> list["Tensor"]:
        """Returns the input tensors saved during forward."""
        raise NotImplementedError("TODO: E05")


class AddBackward(Function):
    """Backward function for element-wise addition: z = a + b.

    Gradients: dz/da = 1, dz/db = 1
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E05")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")


class SubBackward(Function):
    """Backward function for element-wise subtraction: z = a - b.

    Gradients: dz/da = 1, dz/db = -1
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E05")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")


class MulBackward(Function):
    """Backward function for element-wise multiplication: z = a * b.

    Gradients: dz/da = b, dz/db = a
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E05")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")


class DivBackward(Function):
    """Backward function for element-wise division: z = a / b.

    Gradients: dz/da = 1/b, dz/db = -a/b^2
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E05")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E05")


# ================================================================
# E06 — More Backward Ops
# ================================================================


class MatMulBackward(Function):
    """Backward for matrix multiplication: z = a @ b.

    Gradients: dz/da = grad @ b^T, dz/db = a^T @ grad
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class SumBackward(Function):
    """Backward for sum along axis.

    Gradient: broadcast grad back to the original shape.
    """

    def __init__(self, axis: int, keep_dims: bool) -> None:
        raise NotImplementedError("TODO: E06")

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class MeanBackward(Function):
    """Backward for mean along axis.

    Gradient: broadcast grad / axis_length back to the original shape.
    """

    def __init__(self, axis: int, keep_dims: bool) -> None:
        raise NotImplementedError("TODO: E06")

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class ExpBackward(Function):
    """Backward for exp: y = exp(x).

    Gradient: grad * exp(x).
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class LogBackward(Function):
    """Backward for log: y = log(x).

    Gradient: grad / x.
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class ReshapeBackward(Function):
    """Backward for reshape.

    Gradient: reshape grad back to the original shape.
    """

    def __init__(self, *target_shape: int) -> None:
        raise NotImplementedError("TODO: E06")

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class TransposeBackward(Function):
    """Backward for transpose.

    Gradient: transpose grad with the inverse permutation.
    """

    def __init__(self, *axes: int) -> None:
        raise NotImplementedError("TODO: E06")

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class ReLUBackward(Function):
    """Backward for ReLU activation.

    Gradient: grad * (x > 0).
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class SigmoidBackward(Function):
    """Backward for sigmoid activation.

    Gradient: grad * sigmoid(x) * (1 - sigmoid(x)).
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class TanhBackward(Function):
    """Backward for tanh activation.

    Gradient: grad * (1 - tanh(x)^2).
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class GELUBackward(Function):
    """Backward for GELU activation (tanh approximation).

    gelu(x) = 0.5 * x * (1 + tanh(s)), s = sqrt(2/pi) * (x + 0.044715 * x^3)
    gelu'(x) = 0.5*(1+tanh(s)) + 0.5*x*(1-tanh(s)^2)*sqrt(2/pi)*(1+3*0.044715*x^2)
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class CrossEntropyBackward(Function):
    """Backward for cross-entropy loss (fused gradient).

    Gradient: (softmax(logits) - targets) / batch_size
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")


class DropoutBackward(Function):
    """Backward for dropout.

    Gradient: grad * mask (reuses mask from forward pass).
    """

    def forward(self, *inputs: "Tensor") -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")

    def backward(self, grad_output: "NDArray") -> list["NDArray"]:
        raise NotImplementedError("TODO: E06")

    def get_inputs(self) -> list["Tensor"]:
        raise NotImplementedError("TODO: E06")
