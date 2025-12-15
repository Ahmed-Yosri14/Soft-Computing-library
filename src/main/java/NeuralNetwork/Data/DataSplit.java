package NeuralNetwork.Data;

public class DataSplit {

    public static SplitResult split(double[][] X, double[][] y, double ratio) {
        int trainSize = (int) (X.length * ratio);

        double[][] Xtrain = new double[trainSize][X[0].length];
        double[][] ytrain = new double[trainSize][y[0].length];
        double[][] Xtest = new double[X.length - trainSize][X[0].length];
        double[][] ytest = new double[y.length - trainSize][y[0].length];

        for (int i = 0; i < X.length; i++) {
            if (i < trainSize) {
                Xtrain[i] = X[i];
                ytrain[i] = y[i];
            } else {
                Xtest[i - trainSize] = X[i];
                ytest[i - trainSize] = y[i];
            }
        }
        return new SplitResult(Xtrain, ytrain, Xtest, ytest);
    }
}
