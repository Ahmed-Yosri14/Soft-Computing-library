package NeuralNetwork.Data;

public class NormalizationUtils {

    public static class NormalizationStats {
        public double[] min;
        public double[] max;
        public double[] mean;
        public double[] std;

        public NormalizationStats(int features) {
            min = new double[features];
            max = new double[features];
            mean = new double[features];
            std = new double[features];
        }
    }

    public static void minMax(double[][] X) {
        int features = X[0].length;

        for (int j = 0; j < features; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;

            for (double[] row : X) {
                min = Math.min(min, row[j]);
                max = Math.max(max, row[j]);
            }

            for (double[] row : X) {
                if (max - min > 0) {
                    row[j] = (row[j] - min) / (max - min);
                }
            }
        }
    }
}
