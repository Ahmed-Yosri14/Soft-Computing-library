package NeuralNetwork.Data;

public class Metric {

    public static double accuracy(double[][] yTrue, double[][] yPred) {
        int correct = 0;

        for (int i = 0; i < yTrue.length; i++) {

            if (yTrue[i].length == 1) {
                int actual = yTrue[i][0] >= 0.5 ? 1 : 0;
                int predicted = yPred[i][0] >= 0.5 ? 1 : 0;

                if (actual == predicted) correct++;
            }
            else {
                int actual = argMax(yTrue[i]);
                int predicted = argMax(yPred[i]);

                if (actual == predicted) correct++;
            }
        }
        return (double) correct / yTrue.length;
    }

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
