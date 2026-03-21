package com.cognitivediagnosis.models;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for cognitive diagnosis neural networks
 * Corresponds to BaseNet in the Python implementation
 */
public abstract class BaseNet {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseNet.class);
    protected MultiLayerNetwork network;
    protected String modelName;
    
    public BaseNet(String modelName) {
        this.modelName = modelName;
    }
    
    /**
     * Build the network configuration - to be implemented by subclasses
     */
    protected abstract MultiLayerConfiguration buildNetworkConfiguration(int inputSize, int outputSize);
    
    /**
     * Initialize the network with given input and output dimensions
     */
    public void initializeNetwork(int inputSize, int outputSize) {
        MultiLayerConfiguration config = buildNetworkConfiguration(inputSize, outputSize);
        this.network = new MultiLayerNetwork(config);
        this.network.init();
        this.network.setListeners(new ScoreIterationListener(10));
    }
    
    /**
     * Forward pass through the network
     */
    public INDArray forward(INDArray input) {
        return network.output(input);
    }
    
    /**
     * Make predictions, optionally converting to binary
     */
    public INDArray predict(INDArray features, boolean toBinary) {
        INDArray predictions = forward(features);
        if (toBinary) {
            return Transforms.sign(predictions.sub(0.5)).add(1).div(2);
        }
        return predictions;
    }
    
    /**
     * Calculate Pattern Accuracy Rate (PAR)
     * Corresponds to metric_par in Python
     */
    public static double calculatePAR(INDArray predictions, INDArray targets) {
        INDArray binaryPreds = Transforms.sign(predictions.sub(0.5)).add(1).div(2);
        INDArray correct = binaryPreds.eq(targets);
        INDArray patternCorrect = correct.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE).prod(1); // Product along axis 1
        return patternCorrect.meanNumber().doubleValue();
    }
    
    /**
     * Calculate Attribute Accuracy Rate (AAR)
     * Corresponds to metric_acc in Python
     */
    public static double calculateAAR(INDArray predictions, INDArray targets) {
        INDArray binaryPreds = Transforms.sign(predictions.sub(0.5)).add(1).div(2);
        INDArray correct = binaryPreds.eq(targets);
        return correct.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE).meanNumber().doubleValue();
    }
    
    /**
     * Train the network
     */
    public TrainingHistory trainNetwork(DataSetIterator trainIterator, 
                                      DataSetIterator testIterator,
                                      int epochs, 
                                      boolean verbose) {
        
        List<Double> trainLosses = new ArrayList<>();
        List<Double> trainAAR = new ArrayList<>();
        List<Double> trainPAR = new ArrayList<>();
        List<Double> testLosses = new ArrayList<>();
        List<Double> testAAR = new ArrayList<>();
        List<Double> testPAR = new ArrayList<>();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Training phase
            network.fit(trainIterator);
            
            // Evaluate on training set
            trainIterator.reset();
            double trainLoss = network.score();
            
            // Calculate training metrics
            INDArray trainFeatures = null, trainLabels = null;
            trainIterator.reset();
            while (trainIterator.hasNext()) {
                DataSet ds = trainIterator.next();
                if (trainFeatures == null) {
                    trainFeatures = ds.getFeatures();
                    trainLabels = ds.getLabels();
                } else {
                    trainFeatures = Nd4j.vstack(trainFeatures, ds.getFeatures());
                    trainLabels = Nd4j.vstack(trainLabels, ds.getLabels());
                }
            }
            
            INDArray trainPreds = forward(trainFeatures);
            double trainAARValue = calculateAAR(trainPreds, trainLabels);
            double trainPARValue = calculatePAR(trainPreds, trainLabels);
            
            trainLosses.add(trainLoss);
            trainAAR.add(trainAARValue);
            trainPAR.add(trainPARValue);
            
            // Evaluate on test set if provided
            if (testIterator != null) {
                INDArray testFeatures = null, testLabels = null;
                testIterator.reset();
                while (testIterator.hasNext()) {
                    DataSet ds = testIterator.next();
                    if (testFeatures == null) {
                        testFeatures = ds.getFeatures();
                        testLabels = ds.getLabels();
                    } else {
                        testFeatures = Nd4j.vstack(testFeatures, ds.getFeatures());
                        testLabels = Nd4j.vstack(testLabels, ds.getLabels());
                    }
                }
                
                INDArray testPreds = forward(testFeatures);
                double testLoss = network.score(new DataSet(testFeatures, testLabels), false);
                double testAARValue = calculateAAR(testPreds, testLabels);
                double testPARValue = calculatePAR(testPreds, testLabels);
                
                testLosses.add(testLoss);
                testAAR.add(testAARValue);
                testPAR.add(testPARValue);
            }
            
            if (verbose && epoch % Math.max(1, epochs / 20) == 0) {
                String logMessage = String.format("%s Epoch %d: Train Loss=%.4f, Train AAR=%.4f, Train PAR=%.4f", 
                    modelName, epoch, trainLoss, trainAARValue, trainPARValue);
                if (testIterator != null) {
                    logMessage += String.format(", Test Loss=%.4f, Test AAR=%.4f, Test PAR=%.4f", 
                        testLosses.get(testLosses.size()-1), 
                        testAAR.get(testAAR.size()-1), 
                        testPAR.get(testPAR.size()-1));
                }
                logger.info(logMessage);
            }
            
            trainIterator.reset();
            if (testIterator != null) {
                testIterator.reset();
            }
        }
        
        return new TrainingHistory(trainLosses, trainAAR, trainPAR, testLosses, testAAR, testPAR);
    }
    
    /**
     * Get the underlying network
     */
    public MultiLayerNetwork getNetwork() {
        return network;
    }
    
    /**
     * Get model name
     */
    public String getModelName() {
        return modelName;
    }
}
