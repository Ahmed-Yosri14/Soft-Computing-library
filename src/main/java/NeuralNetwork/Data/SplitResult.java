package NeuralNetwork.Data;

// Record to hold the result of train/test split
public record SplitResult(
        double[][] Xtrain,  // Training features
        double[][] ytrain,  // Training labels
        double[][] Xtest,   // Test features
        double[][] ytest    // Test labels
) {}
