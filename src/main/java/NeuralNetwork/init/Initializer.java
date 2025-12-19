package NeuralNetwork.init;

// Interface for weight initialization strategies
public interface Initializer {
    // Initialize weight matrix with appropriate values
    double[][] initialize(int inputSize, int outputSize);
}
