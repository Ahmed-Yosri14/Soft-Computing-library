package NeuralNetwork.Activations;

public class Linear extends Activation {

    @Override
    public double[] forward(double[] input) {
        return input;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        return gradOutput;
    }
}
