package NeuralNetwork.Core;

import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Loss.Loss;
import NeuralNetwork.Optimizer.Optimizer;
import NeuralNetwork.Data.Metric;

import java.util.ArrayList;
import java.util.List;

// Main neural network class that manages layers, training, and prediction
public class NeuralNetwork {

    // List of layers in the network
    private List<Layer> layers = new ArrayList<>();
    // Optimizer for updating weights during training
    private Optimizer optimizer;
    // History of loss values for each epoch
    private List<Double> lossHistory = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    // Forward pass: propagate input through all layers
    public double[] forward(double[] input) {
        for (Layer layer : layers)
            input = layer.forward(input);
        return input;
    }

    // Backward pass: propagate gradients through all layers in reverse
    public void backward(double[] grad) {
        for (int i = layers.size() - 1; i >= 0; i--)
            grad = layers.get(i).backward(grad);
    }

    // Train the network using mini-batch gradient descent
    public void train(double[][] X, double[][] y,
                      Loss lossFn,
                      int epochs,
                      int batchSize) {

        // Iterate through each epoch
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            // Process data in batches
            for (int i = 0; i < X.length; i += batchSize) {

                // Reset gradients for all layers
                for (Layer layer : layers)
                    layer.zeroGrad();

                // Calculate end index for current batch
                int end = Math.min(i + batchSize, X.length);

                // Process each sample in the batch
                for (int j = i; j < end; j++) {
                    double[] yPred = forward(X[j]);
                    totalLoss += lossFn.forward(y[j], yPred);
                    backward(lossFn.backward(y[j], yPred));
                }

                // Update weights using accumulated gradients
                optimizer.updateBatch(layers, end - i);
            }

            // Calculate and store average loss for this epoch
            double avgLoss = totalLoss / X.length;
            lossHistory.add(avgLoss);

            // Print progress
            System.out.println("Epoch " + (epoch + 1) +
                    " | Loss = " + avgLoss);
        }
    }

    // Predict output for a single input
    public double[] predict(double[] x) {
        return forward(x);
    }

    // Predict outputs for multiple inputs
    public double[][] predict(double[][] X) {
        double[][] preds = new double[X.length][];
        for (int i = 0; i < X.length; i++)
            preds[i] = forward(X[i]);
        return preds;
    }

    // Evaluate model accuracy on test data
    public double evaluate(double[][] X, double[][] y) {
        return Metric.accuracy(y, predict(X));
    }

    public List<Double> getLossHistory() {
        return lossHistory;
    }
}
