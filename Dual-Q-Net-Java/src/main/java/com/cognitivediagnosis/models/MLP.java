package com.cognitivediagnosis.models;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Multi-Layer Perceptron for cognitive diagnosis
 * Corresponds to MLP in the Python implementation
 */
public class MLP extends BaseNet {
    
    public MLP() {
        super("MLP");
    }
    
    @Override
    protected MultiLayerConfiguration buildNetworkConfiguration(int inputSize, int outputSize) {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.01))
            .list()
            .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(inputSize)
                .nOut(outputSize)
                .activation(Activation.SIGMOID)
                .build())
            .build();
    }
}
