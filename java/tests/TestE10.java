// TestE10.java — E10 DataLoader & MLP test driver
// Provided by tinytorch-starter. Do NOT modify.

import dev.tensorhero.tinytorch.Tensor;
import dev.tensorhero.tinytorch.DataLoader;
import dev.tensorhero.tinytorch.DataLoader.Batch;
import dev.tensorhero.tinytorch.DataLoader.TensorDataset;
import dev.tensorhero.tinytorch.Linear;
import dev.tensorhero.tinytorch.Activations;
import dev.tensorhero.tinytorch.Losses;
import dev.tensorhero.tinytorch.SGD;
import dev.tensorhero.tinynum.NDArray;

import java.util.ArrayList;
import java.util.List;

public class TestE10 {
    public static void main(String[] args) {
        // =============================================================
        // Test 1: dataset size
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 1, 0}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            emit("dataset_size", String.valueOf(ds.size()));
        }

        // =============================================================
        // Test 2–3: dataset get — values
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 1, 0}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);

            Batch b1 = ds.get(1);
            emit("get1_data0", floatStr(b1.data.data.get(0)));
            emit("get1_data1", floatStr(b1.data.data.get(1)));
            emit("get1_label0", floatStr(b1.labels.data.get(0)));

            Batch b2 = ds.get(2);
            emit("get2_data0", floatStr(b2.data.data.get(0)));
            emit("get2_data1", floatStr(b2.data.data.get(1)));
            emit("get2_label0", floatStr(b2.labels.data.get(0)));
        }

        // =============================================================
        // Test 4: numBatches — even split
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 1, 0}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 2, false);
            emit("num_batches_even", String.valueOf(loader.numBatches()));
        }

        // =============================================================
        // Test 5: numBatches — uneven split (ceil)
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{1,2, 3,4, 5,6, 7,8, 9,10}, 5, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 0, 1, 0}, 5, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 2, false);
            emit("num_batches_uneven", String.valueOf(loader.numBatches()));
        }

        // =============================================================
        // Test 6–7: Batch shapes — no shuffle
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 1, 0}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 2, false);

            List<Batch> batches = new ArrayList<>();
            for (Batch b : loader) {
                batches.add(b);
            }
            emit("batch_count", String.valueOf(batches.size()));

            // First batch shape
            int[] dataShape = batches.get(0).data.data.shape();
            int[] labelShape = batches.get(0).labels.data.shape();
            emit("batch0_data_rows", String.valueOf(dataShape[0]));
            emit("batch0_data_cols", String.valueOf(dataShape[1]));
            emit("batch0_label_rows", String.valueOf(labelShape[0]));
            emit("batch0_label_cols", String.valueOf(labelShape[1]));
        }

        // =============================================================
        // Test 8: Total samples coverage (no shuffle)
        // Verify all data is iterated exactly once
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{10,20, 30,40, 50,60, 70,80}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{1, 2, 3, 4}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 3, false);

            float labelSum = 0;
            int totalSamples = 0;
            for (Batch b : loader) {
                int batchRows = b.labels.data.shape()[0];
                for (int i = 0; i < batchRows; i++) {
                    labelSum += b.labels.data.get(i, 0);
                }
                totalSamples += batchRows;
            }
            emit("total_samples", String.valueOf(totalSamples));
            emit("label_sum", floatStr(labelSum));
        }

        // =============================================================
        // Test 9: Last batch may be smaller
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{1,2, 3,4, 5,6, 7,8, 9,10}, 5, 2);
            NDArray labels = NDArray.fromArray(new float[]{0, 1, 0, 1, 0}, 5, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 2, false);

            List<Batch> batches = new ArrayList<>();
            for (Batch b : loader) {
                batches.add(b);
            }
            int lastBatchSize = batches.get(batches.size() - 1).data.data.shape()[0];
            emit("last_batch_size", String.valueOf(lastBatchSize));
        }

        // =============================================================
        // Test 10: No-shuffle preserves order
        // =============================================================
        {
            NDArray data = NDArray.fromArray(new float[]{10,20, 30,40, 50,60, 70,80}, 4, 2);
            NDArray labels = NDArray.fromArray(new float[]{1, 2, 3, 4}, 4, 1);
            TensorDataset ds = new TensorDataset(data, labels);
            DataLoader loader = new DataLoader(ds, 2, false);

            List<Batch> batches = new ArrayList<>();
            for (Batch b : loader) {
                batches.add(b);
            }
            // batch 0 should have data[[10,20],[30,40]], labels[[1],[2]]
            emit("noshuffle_b0_d00", floatStr(batches.get(0).data.data.get(0, 0)));
            emit("noshuffle_b0_d10", floatStr(batches.get(0).data.data.get(1, 0)));
            emit("noshuffle_b1_d00", floatStr(batches.get(1).data.data.get(0, 0)));
            emit("noshuffle_b1_d10", floatStr(batches.get(1).data.data.get(1, 0)));
        }

        // =============================================================
        // Test 11–12: XOR training integration 🎉
        // Train a 2-layer MLP on XOR for 200 epochs
        // =============================================================
        {
            NDArray.manualSeed(7); // deterministic weight init for reproducible XOR training
            NDArray xorData = NDArray.fromArray(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
            NDArray xorLabels = NDArray.fromArray(new float[]{0, 1, 1, 0}, 4, 1);
            TensorDataset ds = new TensorDataset(xorData, xorLabels);
            DataLoader loader = new DataLoader(ds, 4, false);

            Linear layer1 = new Linear(2, 8);
            Linear layer2 = new Linear(8, 1);

            // Enable gradients
            for (Tensor p : layer1.parameters()) p.requiresGrad = true;
            for (Tensor p : layer2.parameters()) p.requiresGrad = true;

            List<Tensor> allParams = new ArrayList<>();
            allParams.addAll(layer1.parameters());
            allParams.addAll(layer2.parameters());
            SGD optimizer = new SGD(allParams, 0.5f);

            float firstLoss = 0;
            float lastLoss = 0;

            for (int epoch = 0; epoch < 200; epoch++) {
                for (Batch batch : loader) {
                    Tensor h = Activations.relu(layer1.forward(batch.data));
                    Tensor pred = layer2.forward(h);
                    Tensor loss = Losses.mse(pred, batch.labels);
                    loss.backward();
                    optimizer.step();
                    optimizer.zeroGrad();

                    float lossVal = loss.data.get(0);
                    if (epoch == 0) firstLoss = lossVal;
                    lastLoss = lossVal;
                }
            }

            emit("xor_loss_decreases", String.valueOf(lastLoss < firstLoss));

            // Verify predictions
            Tensor xIn = Tensor.fromNDArray(xorData);
            Tensor h = Activations.relu(layer1.forward(xIn));
            Tensor preds = layer2.forward(h);

            int correct = 0;
            for (int i = 0; i < 4; i++) {
                float predicted = preds.data.get(i, 0);
                float expected = xorLabels.get(i, 0);
                if (Math.abs(Math.round(predicted) - expected) < 0.5) correct++;
            }
            emit("xor_accuracy", floatStr((float) correct / 4.0f));
        }
    }

    static void emit(String testName, String result) {
        System.out.println("TEST:" + testName);
        System.out.println("RESULT:" + result);
    }

    static String floatStr(float value) {
        return String.format("%.6f", value);
    }
}
