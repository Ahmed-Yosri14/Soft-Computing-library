package NeuralNetwork.Optimizer;

import NeuralNetwork.Layers.Layer;
import java.util.List;

// Stochastic Gradient Descent optimizer
public class SGD implements Optimizer {

    // Learning rate for weight updates
    private double lr;

    // Initialize SGD with learning rate
    public SGD(double lr) {
        this.lr = lr;
    }

    // Update weights using accumulated gradients divided by batch size
    @Override
    public void updateBatch(List<Layer> layers, int batchSize) {
        // Scale learning rate by batch size for averaging
        double effectiveLR = lr / batchSize;
        for (Layer layer : layers) {
            layer.update(effectiveLR);
        }
    }
}
