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
            // Check if scanner has next token
            if (!sc.hasNext()) {
                System.out.println("No input available. Using default: " + def);
                return def;
            }

            String input = sc.next();
            int v = Integer.parseInt(input);

            if (v < min || v > max) {
                System.out.println("Input out of range [" + min + "-" + max + "]. Using default: " + def);
                return def;
            }
            return v;
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format. Using default: " + def);
            return def;
        } catch (Exception e) {
            System.out.println("Error reading input: " + e.getMessage() + ". Using default: " + def);
            return def;
        } finally {
            // Clear any remaining input on the line
            if (sc.hasNextLine()) {
                sc.nextLine();
            }
        }
    }

    // Read and validate double input within specified range
    private static double readDouble(Scanner sc, String msg, double min, double max, double def) {
        System.out.print(msg);
        try {
            // Check if scanner has next token
            if (!sc.hasNext()) {
                System.out.println("No input available. Using default: " + def);
                return def;
            }

            String input = sc.next();
            double v = Double.parseDouble(input);

            // Check for special values
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                System.out.println("Invalid number (NaN or Infinity). Using default: " + def);
                return def;
            }

            if (v < min || v > max) {
                System.out.println("Input out of range [" + min + "-" + max + "]. Using default: " + def);
                return def;
            }
            return v;
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format. Using default: " + def);
            return def;
        } catch (Exception e) {
            System.out.println("Error reading input: " + e.getMessage() + ". Using default: " + def);
            return def;
        } finally {
            // Clear any remaining input on the line
            if (sc.hasNextLine()) {
                sc.nextLine();
            }
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
    public static void main(String[] args) {
        Scanner scanner = null;

        try {
            scanner = new Scanner(System.in);

            // Display menu options
            System.out.println("=== Neural Network Case Study ===");
            System.out.println("1) Run default Vertebral Column classification");
            System.out.println("2) Fully customize neural network");
            int mode = readInt(scanner, "Choose option: ", 1, 2, 1);

            // ===============================
            // Load dataset
            // ===============================
            // Load vertebral column dataset from CSV file
            List<String[]> rawData;
            try {
                rawData = DataLoader.loadCSV("column_2C.csv");
            } catch (Exception e) {
                System.err.println("Error loading dataset: " + e.getMessage());
                System.err.println("Please ensure 'column_2C.csv' exists in the resources folder.");
                return;
            }

            // Validate dataset
            if (rawData == null || rawData.size() <= 1) {
                System.err.println("Error: Dataset is empty or contains only header.");
                return;
            }

            // Parse CSV data into feature matrix X and label vector y
            int samples = rawData.size() - 1;
            if (samples <= 0) {
                System.err.println("Error: No data samples found in dataset.");
                return;
            }

            double[][] X = new double[samples][6];  // 6 features per sample
            double[][] y = new double[samples][1];  // Binary classification label

            // Convert string data to numerical arrays
            int validSamples = 0;
            for (int i = 0; i < samples; i++) {
                try {
                    String[] row = rawData.get(i + 1);

                    // Validate row has enough columns
                    if (row == null || row.length < 7) {
                        System.out.println("Warning: Skipping incomplete row " + (i + 1));
                        continue;
                    }

                    // Parse feature values
                    for (int j = 0; j < 6; j++) {
                        try {
                            double value = Double.parseDouble(row[j].trim());
                            // Check for invalid values
                            if (Double.isNaN(value) || Double.isInfinite(value)) {
                                X[validSamples][j] = 0.0;
                            } else {
                                X[validSamples][j] = value;
                            }
                        } catch (Exception e) {
                            X[validSamples][j] = 0.0; // Graceful fallback for invalid data
                        }
                    }

                    // Convert label: Abnormal=1, Normal=0
                    if (row[6] != null) {
                        y[validSamples][0] = row[6].trim().equalsIgnoreCase("Abnormal") ? 1.0 : 0.0;
                    } else {
                        y[validSamples][0] = 0.0;
                    }

                    validSamples++;
                } catch (Exception e) {
                    System.out.println("Warning: Error parsing row " + (i + 1) + ": " + e.getMessage());
                }
            }

            // Check if we have enough valid samples
            if (validSamples < 10) {
                System.err.println("Error: Not enough valid samples (" + validSamples + "). Need at least 10.");
                return;
            }

            // Trim arrays to actual valid samples if needed
            if (validSamples < samples) {
                double[][] XTrimmed = new double[validSamples][6];
                double[][] yTrimmed = new double[validSamples][1];
                System.arraycopy(X, 0, XTrimmed, 0, validSamples);
                System.arraycopy(y, 0, yTrimmed, 0, validSamples);
                X = XTrimmed;
                y = yTrimmed;
            }

            System.out.println("Loaded " + validSamples + " valid samples from dataset.");

            // ===============================
            // Split then normalize
            // ===============================
            // Split data into 80% train, 20% test
            SplitResult split;
            try {
                split = DataSplit.split(X, y, 0.8);
            } catch (Exception e) {
                System.err.println("Error splitting dataset: " + e.getMessage());
                return;
            }

            // Validate split results
            if (split.Xtrain() == null || split.Xtrain().length == 0) {
                System.err.println("Error: Training set is empty after split.");
                return;
            }

            if (split.Xtest() == null || split.Xtest().length == 0) {
                System.err.println("Error: Test set is empty after split.");
                return;
            }

            System.out.println("Train samples: " + split.Xtrain().length);
            System.out.println("Test samples: " + split.Xtest().length);

            // Compute normalization statistics from training data only
            try {
                var stats = NormalizationUtils.fitMinMax(split.Xtrain());
                // Apply normalization to both train and test sets
                NormalizationUtils.transformMinMax(split.Xtrain(), stats);
                NormalizationUtils.transformMinMax(split.Xtest(), stats);
            } catch (Exception e) {
                System.err.println("Error during normalization: " + e.getMessage());
                System.err.println("Continuing without normalization...");
            }

            // ===============================
            // Defaults (unchanged)
            // ===============================
            // Default hyperparameters
            int epochs = 500;
            int batchSize = 16;
            double learningRate = 0.01;

            // Adjust batch size if it exceeds training data size
            int maxBatchSize = split.Xtrain().length;
            if (batchSize > maxBatchSize) {
                batchSize = Math.max(1, maxBatchSize / 2);
                System.out.println("Adjusted batch size to " + batchSize + " based on training data size.");
            }

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

                // Calculate reasonable batch size range
                int minBatch = 1;
                int maxBatch = Math.min(128, split.Xtrain().length);

                System.out.println("Note: Training data has " + split.Xtrain().length + " samples.");
                batchSize = readInt(scanner, "Batch size (1–" + maxBatch + "): ",
                        minBatch, maxBatch, Math.min(batchSize, maxBatch));

                // Validate batch size doesn't exceed training data
                if (batchSize > split.Xtrain().length) {
                    batchSize = Math.max(1, split.Xtrain().length / 2);
                    System.out.println("Batch size adjusted to " + batchSize + " (cannot exceed training samples).");
                }

                learningRate = readDouble(scanner, "Learning rate (0.0001–1.0): ",
                        0.0001, 1.0, learningRate);

                // Validate learning rate is reasonable
                if (learningRate <= 0 || learningRate > 1.0) {
                    System.out.println("Invalid learning rate. Using default: 0.01");
                    learningRate = 0.01;
                }

                // Select optimizer (currently only SGD available)
                System.out.println("\nOptimizer:");
                System.out.println("1) SGD");
                readInt(scanner, "Choose: ", 1, 1, 1);

                try {
                    optimizer = new SGD(learningRate);
                } catch (Exception e) {
                    System.err.println("Error creating optimizer: " + e.getMessage());
                    System.out.println("Using default SGD optimizer with LR=0.01");
                    optimizer = new SGD(0.01);
                }

                // Select weight initialization method
                System.out.println("\nInitializer:");
                System.out.println("1) Xavier");
                System.out.println("2) RandomUniform");
                int initChoice = readInt(scanner, "Choose: ", 1, 2, 1);

                try {
                    initializer = (initChoice == 2) ? new RandomUniform() : new Xavier();
                } catch (Exception e) {
                    System.err.println("Error creating initializer: " + e.getMessage());
                    System.out.println("Using default Xavier initializer");
                    initializer = new Xavier();
                }
            }

            // ===============================
            // Build Neural Network
            // ===============================
            NeuralNetwork nn = new NeuralNetwork();

            try {
                if (mode == 2) {
                    // Custom architecture: user defines layers

                    int hiddenLayers = readInt(scanner,
                            "Number of hidden layers (1–10): ", 1, 10, 2);

                    // Validate hidden layers count
                    if (hiddenLayers < 1 || hiddenLayers > 10) {
                        System.out.println("Invalid layer count. Using default: 2");
                        hiddenLayers = 2;
                    }

                    int inputSize = 6;

                    // Build each hidden layer
                    for (int i = 0; i < hiddenLayers; i++) {

                        int neurons = readInt(
                                scanner,
                                "Neurons in hidden layer " + (i + 1) + " (1–256): ",
                                1, 256, 8
                        );

                        // Validate neuron count
                        if (neurons < 1) {
                            System.out.println("Invalid neuron count. Using minimum: 1");
                            neurons = 1;
                        } else if (neurons > 256) {
                            System.out.println("Neuron count too large. Using maximum: 256");
                            neurons = 256;
                        }

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
                        try {
                            nn.addLayer(new Dense(inputSize, neurons, initializer));
                            nn.addLayer(buildActivation(actChoice));
                            inputSize = neurons;
                        } catch (Exception e) {
                            System.err.println("Error creating layer " + (i + 1) + ": " + e.getMessage());
                            throw new RuntimeException("Failed to build network architecture", e);
                        }
                    }

                    // Add output layer for binary classification
                    try {
                        nn.addLayer(new Dense(inputSize, 1, initializer));
                        nn.addLayer(new Sigmoid());
                    } catch (Exception e) {
                        System.err.println("Error creating output layer: " + e.getMessage());
                        throw new RuntimeException("Failed to create output layer", e);
                    }

                } else {
                    // Default architecture: 6 -> 12 -> 6 -> 1
                    try {
                        nn.addLayer(new Dense(6, 12, new Xavier()));
                        nn.addLayer(new ReLU());

                        nn.addLayer(new Dense(12, 6, new Xavier()));
                        nn.addLayer(new ReLU());

                        nn.addLayer(new Dense(6, 1, new Xavier()));
                        nn.addLayer(new Sigmoid());
                    } catch (Exception e) {
                        System.err.println("Error building default architecture: " + e.getMessage());
                        throw new RuntimeException("Failed to create default network", e);
                    }
                }

                nn.setOptimizer(optimizer);

            } catch (Exception e) {
                System.err.println("Critical error building neural network: " + e.getMessage());
                System.err.println("Cannot continue without a valid network.");
                return;
            }

            // ===============================
            // Train
            // ===============================
            System.out.println("\nTraining started...");
            System.out.println("Configuration: epochs=" + epochs + ", batchSize=" + batchSize + ", learningRate=" + learningRate);

            try {
                nn.train(
                        split.Xtrain(),
                        split.ytrain(),
                        lossFn,
                        epochs,
                        batchSize
                );
                System.out.println("Training completed successfully.");
            } catch (OutOfMemoryError e) {
                System.err.println("Out of memory during training!");
                System.err.println("Try reducing: batch size, number of neurons, or number of layers.");
                return;
            } catch (ArithmeticException e) {
                System.err.println("Arithmetic error during training: " + e.getMessage());
                System.err.println("This may be caused by numerical instability.");
                System.err.println("Try: reducing learning rate or changing initialization method.");
                return;
            } catch (Exception e) {
                System.err.println("Error during training: " + e.getMessage());
                e.printStackTrace();
                return;
            }

            // ===============================
            // Evaluate
            // ===============================
            try {
                double accuracy = nn.evaluate(split.Xtest(), split.ytest());

                // Validate accuracy is in reasonable range
                if (Double.isNaN(accuracy) || Double.isInfinite(accuracy)) {
                    System.err.println("Warning: Accuracy calculation produced invalid value (NaN or Infinity).");
                    System.out.println("This may indicate a problem with the model or data.");
                } else {
                    System.out.println("\nTest Accuracy: " + String.format("%.4f", accuracy) +
                                     " (" + String.format("%.2f%%", accuracy * 100) + ")");
                }
            } catch (Exception e) {
                System.err.println("Error during evaluation: " + e.getMessage());
                System.err.println("Continuing to predictions anyway...");
            }

            // ===============================
            // Sample predictions
            // ===============================
            System.out.println("\nSample predictions:");
            try {
                int samplesToShow = Math.min(5, split.Xtest().length);
                for (int i = 0; i < samplesToShow; i++) {
                    try {
                        double[] prediction = nn.predict(split.Xtest()[i]);

                        if (prediction == null || prediction.length == 0) {
                            System.out.println("Sample " + i + ": Error - No prediction returned");
                            continue;
                        }

                        double p = prediction[0];

                        // Check for invalid prediction values
                        if (Double.isNaN(p) || Double.isInfinite(p)) {
                            System.out.println("Sample " + i + ": Error - Invalid prediction (NaN or Infinity)");
                        } else {
                            String label = (p >= 0.5) ? "Abnormal" : "Normal";
                            System.out.println("Sample " + i + ": " + label +
                                             " (confidence: " + String.format("%.4f", Math.abs(p - 0.5) + 0.5) + ")");
                        }
                    } catch (Exception e) {
                        System.out.println("Sample " + i + ": Error - " + e.getMessage());
                    }
                }
            } catch (Exception e) {
                System.err.println("Error during sample predictions: " + e.getMessage());
            }

            System.out.println("\n=== Demo Complete ===");

        } catch (OutOfMemoryError e) {
            System.err.println("\n!!! OUT OF MEMORY ERROR !!!");
            System.err.println("The program ran out of memory. Try:");
            System.err.println("  - Reducing the number of neurons");
            System.err.println("  - Reducing the number of layers");
            System.err.println("  - Reducing the batch size");
            System.err.println("  - Running with more heap memory: java -Xmx2g -cp ...");
        } catch (Exception e) {
            System.err.println("\n!!! UNEXPECTED ERROR !!!");
            System.err.println("Error: " + e.getMessage());
            System.err.println("\nStack trace:");
            e.printStackTrace();
        } finally {
            // Ensure scanner is closed
            if (scanner != null) {
                try {
                    scanner.close();
                } catch (Exception e) {
                    // Ignore errors during cleanup
                }
            }
        }
    }
}
