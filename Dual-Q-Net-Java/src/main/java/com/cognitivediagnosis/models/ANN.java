package com.cognitivediagnosis.models;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Artificial Neural Network for cognitive diagnosis
 * Corresponds to ANN in the Python implementation
 */
public class ANN extends BaseNet {
    
    public ANN() {
        super("ANN");
    }
    
    @Override
    protected MultiLayerConfiguration buildNetworkConfiguration(int inputSize, int outputSize) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.01))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(outputSize)
                .activation(Activation.SIGMOID)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(outputSize)
                .nOut(outputSize)
                .activation(Activation.SIGMOID)
                .build())
            .build();
    }
}
