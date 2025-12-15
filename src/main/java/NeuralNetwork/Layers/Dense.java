package NeuralNetwork.Layers;

import NeuralNetwork.init.Initializer;

public class Dense implements Layer {

    private double[][] weights;
    private double[] bias;

    private double[][] gradW;
    private double[] gradB;

    private double[] input;

    public Dense(int inputSize, int outputSize, Initializer initializer) {
        weights = initializer.initialize(inputSize, outputSize);
        bias = new double[outputSize];

        gradW = new double[inputSize][outputSize];
        gradB = new double[outputSize];
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;
        double[] output = new double[bias.length];

        for (int j = 0; j < bias.length; j++) {
            output[j] = bias[j];
            for (int i = 0; i < input.length; i++) {
                output[j] += input[i] * weights[i][j];
            }
        }
        return output;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[input.length];

        for (int j = 0; j < bias.length; j++) {
            gradB[j] += gradOutput[j];
            for (int i = 0; i < input.length; i++) {
                gradW[i][j] += input[i] * gradOutput[j];
                gradInput[i] += weights[i][j] * gradOutput[j];
            }
        }
        return gradInput;
    }

    @Override
    public void update(double lr) {
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] -= lr * gradW[i][j];

        for (int j = 0; j < bias.length; j++)
            bias[j] -= lr * gradB[j];
    }

    @Override
    public void zeroGrad() {
        for (int i = 0; i < gradW.length; i++)
            for (int j = 0; j < gradW[0].length; j++)
                gradW[i][j] = 0;

        for (int j = 0; j < gradB.length; j++)
            gradB[j] = 0;
    }
}
