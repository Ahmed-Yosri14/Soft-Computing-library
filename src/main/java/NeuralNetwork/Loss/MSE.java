package NeuralNetwork.Loss;

// Mean Squared Error loss: L = sum((y_true - y_pred)^2)
public class MSE implements Loss {

    // Compute MSE loss
    @Override
    public double forward(double[] yTrue, double[] yPred) {
        double sum = 0;

        // Sum squared errors
        for (int i = 0; i < yTrue.length; i++)
            sum += Math.pow(yTrue[i] - yPred[i], 2);

        return sum;
    }

    // Compute gradient: dL/dy_pred = 2(y_pred - y_true) / n
    @Override
    public double[] backward(double[] yTrue, double[] yPred) {
        double[] grad = new double[yTrue.length];

        for (int i = 0; i < yTrue.length; i++)
            grad[i] = 2 * (yPred[i] - yTrue[i]) / yTrue.length;

        return grad;
    }
}
