package NeuralNetwork.Loss;

public class BinaryCrossEntropy implements Loss {

    @Override
    public double forward(double[] yTrue, double[] yPred) {
        double loss = 0;
        double eps = 1e-15;

        for (int i = 0; i < yTrue.length; i++) {
            double y = yTrue[i];
            double p = Math.max(eps, Math.min(1 - eps, yPred[i]));
            loss += -y * Math.log(p) - (1 - y) * Math.log(1 - p);
        }

        return loss;
    }

    @Override
    public double[] backward(double[] yTrue, double[] yPred) {
        double[] grad = new double[yTrue.length];
        double eps = 1e-15;

        for (int i = 0; i < yTrue.length; i++) {
            double y = yTrue[i];
            double p = Math.max(eps, Math.min(1 - eps, yPred[i]));
            grad[i] = (p - y) / (p * (1 - p));
        }
        return grad;
    }
}
