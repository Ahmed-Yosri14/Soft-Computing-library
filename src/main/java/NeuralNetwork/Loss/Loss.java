package NeuralNetwork.Loss;

// Interface for loss functions used in training
public interface Loss {
    // Compute loss value between true and predicted values
    double forward(double[] yTrue, double[] yPred);
    // Compute gradient of loss with respect to predictions
    double[] backward(double[] yTrue, double[] yPred);
}
