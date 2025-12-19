package NeuralNetwork.Data;

import java.util.Random;

// Utility for splitting data into train and test sets
public class DataSplit {

    // Split data into train/test sets with specified ratio (e.g., 0.8 = 80% train)
    public static SplitResult split(double[][] X, double[][] y, double ratio) {

        int n = X.length;
        int trainSize = (int) (n * ratio);

        // Create array of indices for shuffling
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Random rand = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        // Allocate arrays for train and test sets
        double[][] Xtrain = new double[trainSize][X[0].length];
        double[][] ytrain = new double[trainSize][y[0].length];
        double[][] Xtest = new double[n - trainSize][X[0].length];
        double[][] ytest = new double[n - trainSize][y[0].length];

        // Fill training set with first trainSize shuffled samples
        for (int i = 0; i < trainSize; i++) {
            Xtrain[i] = X[indices[i]];
            ytrain[i] = y[indices[i]];
        }

        // Fill test set with remaining samples
        for (int i = trainSize; i < n; i++) {
            Xtest[i - trainSize] = X[indices[i]];
            ytest[i - trainSize] = y[indices[i]];
        }

        return new SplitResult(Xtrain, ytrain, Xtest, ytest);
    }
}
