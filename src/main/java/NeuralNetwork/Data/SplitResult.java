package NeuralNetwork.Data;

public record SplitResult(
        double[][] Xtrain,
        double[][] ytrain,
        double[][] Xtest,
        double[][] ytest
) {}
