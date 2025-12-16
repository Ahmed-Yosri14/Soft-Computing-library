package NeuralNetwork.Data;

import java.util.Random;

public class DataSplit {

    public static SplitResult split(double[][] X, double[][] y, double ratio) {

        int n = X.length;
        int trainSize = (int) (n * ratio);

        // Shuffle indices
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Random rand = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        double[][] Xtrain = new double[trainSize][X[0].length];
        double[][] ytrain = new double[trainSize][y[0].length];
        double[][] Xtest = new double[n - trainSize][X[0].length];
        double[][] ytest = new double[n - trainSize][y[0].length];

        for (int i = 0; i < trainSize; i++) {
            Xtrain[i] = X[indices[i]];
            ytrain[i] = y[indices[i]];
        }

        for (int i = trainSize; i < n; i++) {
            Xtest[i - trainSize] = X[indices[i]];
            ytest[i - trainSize] = y[indices[i]];
        }

        return new SplitResult(Xtrain, ytrain, Xtest, ytest);
    }
}
