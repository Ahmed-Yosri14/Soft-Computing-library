package NeuralNetwork.Layers;

// Interface for the basic operations for a neural network layer
public interface Layer {

    // Perform forward pass through the layer
    double[] forward(double[] input);

    // Perform backward pass and return gradients
    double[] backward(double[] gradOutput);

    // Update layer parameters using learning rate
    void update(double learningRate);

    // Reset accumulated gradients to zero
    default void zeroGrad() {}
}
