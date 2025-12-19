package NeuralNetwork.Loss;

public class BinaryCrossEntropy implements Loss {

    // Compute BCE loss: L = -y*log(p) - (1-y)*log(1-p)
    @Override
    public double forward(double[] yTrue, double[] yPred) {
        double loss = 0;
        double eps = 1e-15; // Small value to prevent log(0)

        for (int i = 0; i < yTrue.length; i++) {
            double y = yTrue[i];
            // Clip predictions to prevent log(0) errors
            double p = Math.max(eps, Math.min(1 - eps, yPred[i]));
            loss += -y * Math.log(p) - (1 - y) * Math.log(1 - p);
        }

        return loss;
    }

    // Compute gradient with respect to predictions
    @Override
    public double[] backward(double[] yTrue, double[] yPred) {
        double[] grad = new double[yTrue.length];
        double eps = 1e-15; // Small value for numerical stability

        for (int i = 0; i < yTrue.length; i++) {
            double y = yTrue[i];
            // Clip predictions for numerical stability
            double p = Math.max(eps, Math.min(1 - eps, yPred[i]));
            grad[i] = (p - y) / (p * (1 - p));
        }
        return grad;
    }
}
