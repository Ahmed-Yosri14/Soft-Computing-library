package NeuralNetwork.Core;

import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Loss.Loss;
import NeuralNetwork.Optimizer.Optimizer;
import NeuralNetwork.Data.Metric;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private List<Layer> layers = new ArrayList<>();
    private Optimizer optimizer;
    private List<Double> lossHistory = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public double[] forward(double[] input) {
        for (Layer layer : layers)
            input = layer.forward(input);
        return input;
    }

    public void backward(double[] grad) {
        for (int i = layers.size() - 1; i >= 0; i--)
            grad = layers.get(i).backward(grad);
    }

    public void train(double[][] X, double[][] y,
                      Loss lossFn,
                      int epochs,
                      int batchSize) {

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < X.length; i += batchSize) {

                for (Layer layer : layers)
                    layer.zeroGrad();

                int end = Math.min(i + batchSize, X.length);

                for (int j = i; j < end; j++) {
                    double[] yPred = forward(X[j]);
                    totalLoss += lossFn.forward(y[j], yPred);
                    backward(lossFn.backward(y[j], yPred));
                }

                optimizer.updateBatch(layers, end - i);
            }

            double avgLoss = totalLoss / X.length;
            lossHistory.add(avgLoss);

            System.out.println("Epoch " + (epoch + 1) +
                    " | Loss = " + avgLoss);
        }
    }

    public double[] predict(double[] x) {
        return forward(x);
    }

    public double[][] predict(double[][] X) {
        double[][] preds = new double[X.length][];
        for (int i = 0; i < X.length; i++)
            preds[i] = forward(X[i]);
        return preds;
    }

    public double evaluate(double[][] X, double[][] y) {
        return Metric.accuracy(y, predict(X));
    }

    public List<Double> getLossHistory() {
        return lossHistory;
    }
}
