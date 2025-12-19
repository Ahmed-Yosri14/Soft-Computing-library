package NeuralNetwork.Data;

public class NormalizationUtils {

    public static class MinMaxStats {
        public double[] min;  // Minimum value per feature
        public double[] max;  // Maximum value per feature

        public MinMaxStats(int features) {
            min = new double[features];
            max = new double[features];
        }
    }

    // Compute min/max statistics from training data
    public static MinMaxStats fitMinMax(double[][] X) {
        int features = X[0].length;
        MinMaxStats stats = new MinMaxStats(features);

        // Find min and max for each feature
        for (int j = 0; j < features; j++) {
            stats.min[j] = Double.MAX_VALUE;
            stats.max[j] = Double.MIN_VALUE;

            for (double[] row : X) {
                stats.min[j] = Math.min(stats.min[j], row[j]);
                stats.max[j] = Math.max(stats.max[j], row[j]);
            }
        }
        return stats;
    }

    // Normalize data to [0, 1] range using precomputed statistics
    public static void transformMinMax(double[][] X, MinMaxStats stats) {
        int features = X[0].length;

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < features; j++) {
                // Avoid division by zero
                if (stats.max[j] - stats.min[j] != 0) {
                    X[i][j] = (X[i][j] - stats.min[j]) /
                            (stats.max[j] - stats.min[j]);
                }
            }
        }
    }
}
