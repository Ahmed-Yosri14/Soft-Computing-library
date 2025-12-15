package NeuralNetwork.Optimizer;

import NeuralNetwork.Layers.Layer;
import java.util.List;

public class SGD implements Optimizer {

    private double lr;

    public SGD(double lr) {
        this.lr = lr;
    }

    @Override
    public void updateBatch(List<Layer> layers, int batchSize) {
        double effectiveLR = lr / batchSize;
        for (Layer layer : layers) {
            layer.update(effectiveLR);
        }
    }
}
