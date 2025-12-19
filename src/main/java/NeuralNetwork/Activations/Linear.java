package NeuralNetwork.Activations;

// Linear (Identity) activation: f(x) = x
public class Linear extends Activation {

    // Pass input through unchanged
    @Override
    public double[] forward(double[] input) {
        return input;
    }

    // Gradient is 1, so pass gradients through unchanged
    @Override
    public double[] backward(double[] gradOutput) {
        return gradOutput;
    }
}
