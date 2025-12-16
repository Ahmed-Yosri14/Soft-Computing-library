package NeuralNetwork.Data;

public class NormalizationUtils {

    public static class MinMaxStats {
        public double[] min;
        public double[] max;

        public MinMaxStats(int features) {
            min = new double[features];
            max = new double[features];
        }
    }

    // Compute stats from TRAINING data only
    public static MinMaxStats fitMinMax(double[][] X) {
        int features = X[0].length;
        MinMaxStats stats = new MinMaxStats(features);

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

    // Apply normalization using precomputed stats
    public static void transformMinMax(double[][] X, MinMaxStats stats) {
        int features = X[0].length;

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < features; j++) {
                if (stats.max[j] - stats.min[j] != 0) {
                    X[i][j] = (X[i][j] - stats.min[j]) /
                            (stats.max[j] - stats.min[j]);
                }
            }
        }
    }
}
