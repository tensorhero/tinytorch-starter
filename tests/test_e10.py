"""test_e10.py — E10 DataLoader & MLP test driver.

Provided by tinytorch-starter. Do NOT modify.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import random

from tinynum import NDArray
from tinytorch import Tensor
from tinytorch.dataloader import Batch, TensorDataset, DataLoader
from tinytorch.layer import Linear
from tinytorch import activations, losses
from tinytorch.optimizer import SGD


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def float_str(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    # =============================================================
    # Test 1: dataset size
    # =============================================================
    data = NDArray.from_array([0,0, 0,1, 1,0, 1,1], 4, 2)
    labels = NDArray.from_array([0, 1, 1, 0], 4, 1)
    ds = TensorDataset(data, labels)
    emit("dataset_size", str(ds.size()))

    # =============================================================
    # Test 2–3: dataset get — values
    # =============================================================
    b1 = ds.get(1)
    emit("get1_data0", float_str(b1.data.data.get(0)))
    emit("get1_data1", float_str(b1.data.data.get(1)))
    emit("get1_label0", float_str(b1.labels.data.get(0)))

    b2 = ds.get(2)
    emit("get2_data0", float_str(b2.data.data.get(0)))
    emit("get2_data1", float_str(b2.data.data.get(1)))
    emit("get2_label0", float_str(b2.labels.data.get(0)))

    # =============================================================
    # Test 4: num_batches — even split
    # =============================================================
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    emit("num_batches_even", str(loader.num_batches()))

    # =============================================================
    # Test 5: num_batches — uneven split (ceil)
    # =============================================================
    data5 = NDArray.from_array([1,2, 3,4, 5,6, 7,8, 9,10], 5, 2)
    labels5 = NDArray.from_array([0, 1, 0, 1, 0], 5, 1)
    ds5 = TensorDataset(data5, labels5)
    loader5 = DataLoader(ds5, batch_size=2, shuffle=False)
    emit("num_batches_uneven", str(loader5.num_batches()))

    # =============================================================
    # Test 6–7: Batch shapes — no shuffle
    # =============================================================
    loader2 = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader2)
    emit("batch_count", str(len(batches)))

    data_shape = batches[0].data.data.get_shape()
    label_shape = batches[0].labels.data.get_shape()
    emit("batch0_data_rows", str(data_shape[0]))
    emit("batch0_data_cols", str(data_shape[1]))
    emit("batch0_label_rows", str(label_shape[0]))
    emit("batch0_label_cols", str(label_shape[1]))

    # =============================================================
    # Test 8: Total samples coverage (no shuffle)
    # =============================================================
    data8 = NDArray.from_array([10,20, 30,40, 50,60, 70,80], 4, 2)
    labels8 = NDArray.from_array([1, 2, 3, 4], 4, 1)
    ds8 = TensorDataset(data8, labels8)
    loader8 = DataLoader(ds8, batch_size=3, shuffle=False)

    label_sum = 0.0
    total_samples = 0
    for b in loader8:
        batch_rows = b.labels.data.get_shape()[0]
        for i in range(batch_rows):
            label_sum += b.labels.data.get(i, 0)
        total_samples += batch_rows
    emit("total_samples", str(total_samples))
    emit("label_sum", float_str(label_sum))

    # =============================================================
    # Test 9: Last batch may be smaller
    # =============================================================
    loader9 = DataLoader(ds5, batch_size=2, shuffle=False)
    batches9 = list(loader9)
    last_batch_size = batches9[-1].data.data.get_shape()[0]
    emit("last_batch_size", str(last_batch_size))

    # =============================================================
    # Test 10: No-shuffle preserves order
    # =============================================================
    loader10 = DataLoader(ds8, batch_size=2, shuffle=False)
    batches10 = list(loader10)
    emit("noshuffle_b0_d00", float_str(batches10[0].data.data.get(0, 0)))
    emit("noshuffle_b0_d10", float_str(batches10[0].data.data.get(1, 0)))
    emit("noshuffle_b1_d00", float_str(batches10[1].data.data.get(0, 0)))
    emit("noshuffle_b1_d10", float_str(batches10[1].data.data.get(1, 0)))

    # =============================================================
    # Test 11–12: XOR training integration 🎉
    # =============================================================
    random.seed(7)  # deterministic weight init for reproducible XOR training
    xor_data = NDArray.from_array([0,0, 0,1, 1,0, 1,1], 4, 2)
    xor_labels = NDArray.from_array([0, 1, 1, 0], 4, 1)
    ds_xor = TensorDataset(xor_data, xor_labels)
    loader_xor = DataLoader(ds_xor, batch_size=4, shuffle=False)

    layer1 = Linear(2, 8)
    layer2 = Linear(8, 1)

    # Enable gradients
    for p in layer1.parameters():
        p.requires_grad = True
    for p in layer2.parameters():
        p.requires_grad = True

    all_params = layer1.parameters() + layer2.parameters()
    optimizer = SGD(all_params, lr=0.5)

    first_loss = 0.0
    last_loss = 0.0

    for epoch in range(200):
        for batch in loader_xor:
            h = activations.relu(layer1.forward(batch.data))
            pred = layer2.forward(h)
            loss = losses.mse(pred, batch.labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.data.get(0)
            if epoch == 0:
                first_loss = loss_val
            last_loss = loss_val

    emit("xor_loss_decreases", str(last_loss < first_loss).lower())

    # Verify predictions
    x_in = Tensor.from_ndarray(xor_data)
    h = activations.relu(layer1.forward(x_in))
    preds = layer2.forward(h)

    correct = 0
    for i in range(4):
        predicted = preds.data.get(i, 0)
        expected = xor_labels.get(i, 0)
        if abs(round(predicted) - expected) < 0.5:
            correct += 1
    emit("xor_accuracy", float_str(correct / 4.0))


if __name__ == "__main__":
    main()
