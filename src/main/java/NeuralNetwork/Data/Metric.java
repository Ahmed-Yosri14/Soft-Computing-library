package NeuralNetwork.Data;

// Utility class for computing evaluation metrics
public class Metric {

    // Calculate accuracy: fraction of correct predictions
    public static double accuracy(double[][] yTrue, double[][] yPred) {
        int correct = 0;

        for (int i = 0; i < yTrue.length; i++) {

            // Binary classification: threshold at 0.5
            if (yTrue[i].length == 1) {
                int actual = yTrue[i][0] >= 0.5 ? 1 : 0;
                int predicted = yPred[i][0] >= 0.5 ? 1 : 0;

                if (actual == predicted) correct++;
            }
            // Multi-class classification: use argmax
            else {
                int actual = argMax(yTrue[i]);
                int predicted = argMax(yPred[i]);

                if (actual == predicted) correct++;
            }
        }
        // Return fraction of correct predictions
        return (double) correct / yTrue.length;
    }

    // Find index of maximum value in array
    private static int argMax(double[] arr) {
        int idx = 0;
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}
