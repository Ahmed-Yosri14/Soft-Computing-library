package NeuralNetwork.Optimizer;

import NeuralNetwork.Layers.Layer;
import java.util.List;

public interface Optimizer {
    void updateBatch(List<Layer> layers, int batchSize);
}
