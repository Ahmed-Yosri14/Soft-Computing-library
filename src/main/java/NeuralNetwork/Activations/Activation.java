package NeuralNetwork.Activations;

import NeuralNetwork.Layers.Layer;

// Base class for activation function layers
public abstract class Activation implements Layer {

    // Activation layers have no parameters to update
    @Override
    public void update(double learningRate) {
        // No parameters
    }

    // Activation layers have no gradients to reset
    @Override
    public void zeroGrad() {
        // Nothing to reset
    }
}
