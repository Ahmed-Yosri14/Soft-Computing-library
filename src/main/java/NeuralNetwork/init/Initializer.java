package NeuralNetwork.init;

// Interface for weight initialization strategies
public interface Initializer {
    double[][] initialize(int inputSize, int outputSize);
}
