package NeuralNetwork.Activations;

// ReLU (Rectified Linear Unit) activation: f(x) = max(0, x)
public class ReLU extends Activation {
    // Mask to remember which elements were positive during forward pass
    private boolean[] mask;

    // Apply ReLU: output 0 for negative values, input for positive values
    @Override
    public double[] forward(double[] input) {
        mask = new boolean[input.length];
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            mask[i] = input[i] > 0;
            output[i] = mask[i] ? input[i] : 0;
        }
        return output;
    }

    // Gradient is 1 for positive inputs, 0 for negative inputs
    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = mask[i] ? gradOutput[i] : 0;
        }
        return gradInput;
    }
}
