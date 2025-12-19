package NeuralNetwork.Optimizer;

import NeuralNetwork.Layers.Layer;
import java.util.List;

// Interface for optimization algorithms that update network weights
public interface Optimizer {
    // Update all layer parameters after processing a batch
    void updateBatch(List<Layer> layers, int batchSize);
}
