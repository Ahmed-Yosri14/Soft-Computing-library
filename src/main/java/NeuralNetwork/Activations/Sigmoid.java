package NeuralNetwork.Activations;

// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
public class Sigmoid extends Activation {
    // Cached output from forward pass for use in backward pass
    private double[] output;

    // Apply sigmoid function to scale values between 0 and 1
    @Override
    public double[] forward(double[] input) {
        output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }
        return output;
    }

    // Gradient: f'(x) = f(x) * (1 - f(x))
    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = gradOutput[i] * output[i] * (1 - output[i]);
        }
        return gradInput;
    }
}
