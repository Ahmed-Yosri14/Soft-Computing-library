package NeuralNetwork.init;

import java.util.Random;

// Random uniform initialization in range [-0.5, 0.5]
public class RandomUniform implements Initializer {
    private Random random = new Random();

    // Initialize weights with random values uniformly distributed
    @Override
    public double[][] initialize(int inputSize, int outputSize) {
        double[][] weights = new double[inputSize][outputSize];
        // Sample from uniform distribution [-0.5, 0.5]
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
        }
        return weights;
    }
}