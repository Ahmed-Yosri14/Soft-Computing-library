package NeuralNetwork.init;

import java.util.Random;

// Xavier/Glorot initialization for better gradient flow
public class Xavier implements Initializer {
    private Random random = new Random();

    // Initialize weights uniformly in range [-limit, limit]
    @Override
    public double[][] initialize(int inputSize, int outputSize) {
        // Calculate limit based on layer dimensions
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        double[][] weights = new double[inputSize][outputSize];
        // Sample from uniform distribution
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextDouble() * 2 * limit - limit;
            }
        }
        return weights;
    }
}
