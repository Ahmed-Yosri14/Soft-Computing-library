package NeuralNetwork.Activations;

// Tanh activation: f(x) = tanh(x)
public class Tanh extends Activation {
    // Cached output from forward pass for use in backward pass
    private double[] output;

    // Apply tanh function to scale values between -1 and 1
    @Override
    public double[] forward(double[] input) {
        output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.tanh(input[i]);
        }
        return output;
    }

    // Gradient: f'(x) = 1 - f(x)^2
    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = gradOutput[i] * (1 - output[i] * output[i]);
        }
        return gradInput;
    }
}
