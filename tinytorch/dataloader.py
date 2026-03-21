"""DataLoader — mini-batch iteration over a dataset.

Contains:
- Batch: simple container for (data, labels) Tensor pair.
- TensorDataset: stores data and labels as 2D NDArrays.
- DataLoader: iterates over a TensorDataset in batches.
"""

from __future__ import annotations

from tinynum import NDArray, Slice
from tinytorch.tensor import Tensor


class Batch:
    """A single batch of data and labels."""

    def __init__(self, data: Tensor, labels: Tensor) -> None:
        self.data = data
        self.labels = labels


class TensorDataset:
    """A dataset backed by two 2D NDArrays (data and labels).

    The first dimension of both arrays must match (number of samples).
    ``get(i)`` returns a Batch containing the i-th row of each array.
    """

    # ================================================================
    # E10 — DataLoader & MLP
    # ================================================================

    def __init__(self, data: NDArray, labels: NDArray) -> None:
        raise NotImplementedError("TODO: E10")

    def size(self) -> int:
        """Returns the number of samples."""
        raise NotImplementedError("TODO: E10")

    def get(self, index: int) -> Batch:
        """Returns the i-th sample as a Batch of 1D Tensors."""
        raise NotImplementedError("TODO: E10")


class DataLoader:
    """Iterates over a TensorDataset in mini-batches.

    Supports optional shuffling. The last batch may be smaller than batch_size.
    """

    def __init__(self, dataset: TensorDataset, batch_size: int, shuffle: bool = False) -> None:
        raise NotImplementedError("TODO: E10")

    def num_batches(self) -> int:
        """Returns the number of batches (ceil division)."""
        raise NotImplementedError("TODO: E10")

    def __iter__(self):
        """Yields Batch objects. Shuffles indices if shuffle=True."""
        raise NotImplementedError("TODO: E10")
