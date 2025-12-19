package NeuralNetwork.Layers;

import NeuralNetwork.init.Initializer;

// Fully connected (dense) layer implementation
public class Dense implements Layer {

    // Weight matrix connecting inputs to outputs
    private double[][] weights;
    // Bias vector for each output neuron
    private double[] bias;

    // Accumulated gradients for weights
    private double[][] gradW;
    // Accumulated gradients for biases
    private double[] gradB;

    // Cached input from forward pass (used in backward pass)
    private double[] input;

    // Initialize dense layer with specified dimensions
    public Dense(int inputSize, int outputSize, Initializer initializer) {
        weights = initializer.initialize(inputSize, outputSize);
        bias = new double[outputSize];

        gradW = new double[inputSize][outputSize];
        gradB = new double[outputSize];
    }

    // Compute output: y = Wx + b
    @Override
    public double[] forward(double[] input) {
        this.input = input;
        double[] output = new double[bias.length];

        // For each output neuron
        for (int j = 0; j < bias.length; j++) {
            output[j] = bias[j];
            // Sum weighted inputs
            for (int i = 0; i < input.length; i++) {
                output[j] += input[i] * weights[i][j];
            }
        }
        return output;
    }

    // Compute gradients and propagate error backwards
    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[input.length];

        // For each output neuron
        for (int j = 0; j < bias.length; j++) {
            // Accumulate bias gradient
            gradB[j] += gradOutput[j];
            // For each input
            for (int i = 0; i < input.length; i++) {
                // Accumulate weight gradient
                gradW[i][j] += input[i] * gradOutput[j];
                // Propagate gradient to input
                gradInput[i] += weights[i][j] * gradOutput[j];
            }
        }
        return gradInput;
    }

    // Update weights and biases using gradients
    @Override
    public void update(double lr) {
        // Update weights
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] -= lr * gradW[i][j];

        // Update biases
        for (int j = 0; j < bias.length; j++)
            bias[j] -= lr * gradB[j];
    }

    // Reset all accumulated gradients to zero
    @Override
    public void zeroGrad() {
        // Zero weight gradients
        for (int i = 0; i < gradW.length; i++)
            for (int j = 0; j < gradW[0].length; j++)
                gradW[i][j] = 0;

        // Zero bias gradients
        for (int j = 0; j < gradB.length; j++)
            gradB[j] = 0;
    }
}
