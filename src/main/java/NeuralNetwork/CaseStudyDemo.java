package NeuralNetwork;

import NeuralNetwork.Core.NeuralNetwork;
import NeuralNetwork.Layers.Dense;
import NeuralNetwork.Activations.ReLU;
import NeuralNetwork.Activations.Sigmoid;
import NeuralNetwork.Loss.MSE;
import NeuralNetwork.init.Xavier;
import NeuralNetwork.Optimizer.SGD;
import NeuralNetwork.Data.DataLoader;
import NeuralNetwork.Data.DataSplit;
import NeuralNetwork.Data.NormalizationUtils;
import NeuralNetwork.Data.Metric;

import java.util.List;

public class CaseStudyDemo {

    public static void main(String[] args) throws Exception {

        // 1. Load CSV data
        List<String[]> rawData = DataLoader.loadCSV("column_2C.csv");

        int samples = rawData.size() - 1; // skip header
        double[][] X = new double[samples][6];
        double[][] y = new double[samples][1];

        // 2. Parse data
        for (int i = 0; i < samples; i++) {
            String[] row = rawData.get(i + 1);

            for (int j = 0; j < 6; j++) {
                X[i][j] = Double.parseDouble(row[j]);
            }

            // Binary encoding: Abnormal = 1, Normal = 0
            y[i][0] = row[6].equalsIgnoreCase("Abnormal") ? 1.0 : 0.0;
        }

        // 3. Normalize features
        NormalizationUtils.minMax(X);

        // 4. Train / Test split
        var split = DataSplit.split(X, y, 0.8);

        // 5. Build Neural Network
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new Dense(6, 12, new Xavier()));
        nn.addLayer(new ReLU());

        nn.addLayer(new Dense(12, 6, new Xavier()));
        nn.addLayer(new ReLU());

        nn.addLayer(new Dense(6, 1, new Xavier()));
        nn.addLayer(new Sigmoid());

        // 6. Set optimizer
        nn.setOptimizer(new SGD(0.01));

        // 7. Train
        nn.train(
                split.Xtrain(),
                split.ytrain(),
                new MSE(),
                500,      // epochs
                16        // batch size
        );

        // 8. Evaluate
        double accuracy = nn.evaluate(split.Xtest(), split.ytest());
        System.out.println("Test Accuracy: " + accuracy);

        // 9. Single sample prediction
        double[] sample = split.Xtest()[0];
        double prediction = nn.predict(sample)[0];

        System.out.println("Sample Prediction: " +
                (prediction >= 0.5 ? "Abnormal" : "Normal"));
    }
}
