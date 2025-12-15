package NeuralNetwork.Activations;

import NeuralNetwork.Layers.Layer;

public abstract class Activation implements Layer {

    @Override
    public void update(double learningRate) {
        // No parameters
    }

    @Override
    public void zeroGrad() {
        // Nothing to reset
    }
}
