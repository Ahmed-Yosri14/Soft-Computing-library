package NeuralNetwork;

import NeuralNetwork.Core.NeuralNetwork;
import NeuralNetwork.Layers.Dense;
import NeuralNetwork.Activations.*;
import NeuralNetwork.Loss.*;
import NeuralNetwork.init.*;
import NeuralNetwork.Optimizer.*;
import NeuralNetwork.Data.*;

import java.util.List;
import java.util.Scanner;

public class CaseStudyDemo {

    // ===============================
    // Input validation helpers
    // ===============================
    // Read and validate integer input within specified range
    private static int readInt(Scanner sc, String msg, int min, int max, int def) {
        System.out.print(msg);
        try {
            int v = sc.nextInt();
            if (v < min || v > max) {
                System.out.println("Invalid input. Using default: " + def);
                return def;
            }
            return v;
        } catch (Exception e) {
            sc.nextLine();
            System.out.println("Invalid input. Using default: " + def);
            return def;
        }
    }

    // Read and validate double input within specified range
    private static double readDouble(Scanner sc, String msg, double min, double max, double def) {
        System.out.print(msg);
        try {
            double v = sc.nextDouble();
            if (v < min || v > max) {
                System.out.println("Invalid input. Using default: " + def);
                return def;
            }
            return v;
        } catch (Exception e) {
            sc.nextLine();
            System.out.println("Invalid input. Using default: " + def);
            return def;
        }
    }

    // Create activation layer based on user choice
    private static Activation buildActivation(int choice) {
        return switch (choice) {
            case 2 -> new Sigmoid();
            case 3 -> new Tanh();
            case 4 -> new Linear();
            default -> new ReLU();
        };
    }

    // ===============================
    // Main
    // ===============================
    // Main entry point for neural network case study demo
    public static void main(String[] args) throws Exception {

        Scanner scanner = new Scanner(System.in);

        // Display menu options
        System.out.println("=== Neural Network Case Study ===");
        System.out.println("1) Run default Vertebral Column classification");
        System.out.println("2) Fully customize neural network");
        int mode = readInt(scanner, "Choose option: ", 1, 2, 1);

        // ===============================
        // Load dataset
        // ===============================
        // Load vertebral column dataset from CSV file
        List<String[]> rawData = DataLoader.loadCSV("column_2C.csv");

        // Parse CSV data into feature matrix X and label vector y
        int samples = rawData.size() - 1;
        double[][] X = new double[samples][6];  // 6 features per sample
        double[][] y = new double[samples][1];  // Binary classification label

        // Convert string data to numerical arrays
        for (int i = 0; i < samples; i++) {
            String[] row = rawData.get(i + 1);
            // Parse feature values
            for (int j = 0; j < 6; j++) {
                try {
                    X[i][j] = Double.parseDouble(row[j]);
                } catch (Exception e) {
                    X[i][j] = 0.0; // Graceful fallback for invalid data
                }
            }
            // Convert label: Abnormal=1, Normal=0
            y[i][0] = row[6].equalsIgnoreCase("Abnormal") ? 1.0 : 0.0;
        }

        // ===============================
        // Split then normalize
        // ===============================
        // Split data into 80% train, 20% test
        SplitResult split = DataSplit.split(X, y, 0.8);

        // Compute normalization statistics from training data only
        var stats = NormalizationUtils.fitMinMax(split.Xtrain());
        // Apply normalization to both train and test sets
        NormalizationUtils.transformMinMax(split.Xtrain(), stats);
        NormalizationUtils.transformMinMax(split.Xtest(), stats);

        System.out.println("Train samples: " + split.Xtrain().length);
        System.out.println("Test samples: " + split.Xtest().length);

        // ===============================
        // Defaults (unchanged)
        // ===============================
        // Default hyperparameters
        int epochs = 500;
        int batchSize = 16;
        double learningRate = 0.01;

        // Default components for binary classification
        Loss lossFn = new BinaryCrossEntropy();
        Initializer initializer = new Xavier();
        Optimizer optimizer = new SGD(learningRate);

        // ===============================
        // Customization
        // ===============================
        // Allow user to customize hyperparameters and architecture
        if (mode == 2) {

            // Get training hyperparameters
            epochs = readInt(scanner, "Epochs (1–5000): ", 1, 5000, epochs);
            batchSize = readInt(scanner, "Batch size (1–128): ", 1, 128, batchSize);
            learningRate = readDouble(scanner, "Learning rate (0.0001–1.0): ",
                    0.0001, 1.0, learningRate);

            // Select optimizer (currently only SGD available)
            System.out.println("Optimizer:");
            System.out.println("1) SGD");
            readInt(scanner, "Choose: ", 1, 1, 1);
            optimizer = new SGD(learningRate);

            // Select weight initialization method
            System.out.println("Initializer:");
            System.out.println("1) Xavier");
            System.out.println("2) RandomUniform");
            int initChoice = readInt(scanner, "Choose: ", 1, 2, 1);
            initializer = (initChoice == 2) ? new RandomUniform() : new Xavier();
        }

        // ===============================
        // Build Neural Network
        // ===============================
        NeuralNetwork nn = new NeuralNetwork();

        if (mode == 2) {
            // Custom architecture: user defines layers

            int hiddenLayers = readInt(scanner,
                    "Number of hidden layers (1–10): ", 1, 10, 2);
            int inputSize = 6;

            // Build each hidden layer
            for (int i = 0; i < hiddenLayers; i++) {

                int neurons = readInt(
                        scanner,
                        "Neurons in hidden layer " + (i + 1) + " (1–256): ",
                        1, 256, 8
                );

                // Select activation function for this layer
                System.out.println("""
                        Activation:
                        1) ReLU
                        2) Sigmoid
                        3) Tanh
                        4) Linear
                        """);

                int actChoice = readInt(scanner, "Choose: ", 1, 4, 1);

                // Add dense layer followed by activation
                nn.addLayer(new Dense(inputSize, neurons, initializer));
                nn.addLayer(buildActivation(actChoice));

                inputSize = neurons;
            }

            // Add output layer for binary classification
            nn.addLayer(new Dense(inputSize, 1, initializer));
            nn.addLayer(new Sigmoid());

        } else {
            // Default architecture: 6 -> 12 -> 6 -> 1
            nn.addLayer(new Dense(6, 12, new Xavier()));
            nn.addLayer(new ReLU());

            nn.addLayer(new Dense(12, 6, new Xavier()));
            nn.addLayer(new ReLU());

            nn.addLayer(new Dense(6, 1, new Xavier()));
            nn.addLayer(new Sigmoid());
        }

        nn.setOptimizer(optimizer);

        // ===============================
        // Train
        // ===============================
        System.out.println("\nTraining started...");
        nn.train(
                split.Xtrain(),
                split.ytrain(),
                lossFn,
                epochs,
                batchSize
        );

        // ===============================
        // Evaluate
        // ===============================
        double accuracy = nn.evaluate(split.Xtest(), split.ytest());
        System.out.println("\nTest Accuracy (All Samples): " + accuracy);

        // ===============================
        // Sample predictions
        // ===============================
        System.out.println("\nSample predictions:");
        for (int i = 0; i < Math.min(5, split.Xtest().length); i++) {
            double p = nn.predict(split.Xtest()[i])[0];
            System.out.println(
                    "Sample " + i + ": " + (p >= 0.5 ? "Abnormal" : "Normal")
            );
        }

        scanner.close();
    }
}
