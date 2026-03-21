package com.cognitivediagnosis.examples;

import com.cognitivediagnosis.config.ModelConfig;
import com.cognitivediagnosis.data.DataProcessor;
import com.cognitivediagnosis.models.QNet;
import com.cognitivediagnosis.models.TrainingHistory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Quick start example demonstrating different usage scenarios
 */
public class QuickStartExample {
    
    private static final Logger logger = LoggerFactory.getLogger(QuickStartExample.class);
    
    public static void main(String[] args) {
        // Example 1: Basic usage with default configuration
        basicUsageExample();
        
        // Example 2: Custom configuration
        customConfigExample();
        
        // Example 3: Different Q-matrix sizes
        differentQMatrixExample();
    }
    
    /**
     * Example 1: Basic usage with default settings
     */
    public static void basicUsageExample() {
        logger.info("=== Basic Usage Example ===");
        
        try {
            // Load Q-matrix and generate Q-star
            INDArray qMatrix = DataProcessor.getSampleQMatrix1();
            INDArray qApos = DataProcessor.generateQApos(2, (int) qMatrix.shape()[1]);
            INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
            
            // Create model
            QNet qnet = new QNet(qMatrix, qStar);
            qnet.initializeNetwork((int) qMatrix.shape()[0], (int) qMatrix.shape()[1]);
            
            // Generate data
            DataProcessor.DataSplit data = DataProcessor.generateSyntheticData(qMatrix, 200, 123L);
            
            // Create iterators
            DataSetIterator trainIter = DataProcessor.createDataSetIterator(
                data.getTrainFeatures(), data.getTrainLabels(), 32, true);
            DataSetIterator testIter = DataProcessor.createDataSetIterator(
                data.getTestFeatures(), data.getTestLabels(), 32, false);
            
            // Train with default settings
            TrainingHistory history = qnet.trainNetwork(trainIter, testIter, 50, true);
            
            // Print results
            System.out.println("Basic Example Results:");
            history.printSummary();
            
        } catch (Exception e) {
            logger.error("Error in basic example: ", e);
        }
    }
    
    /**
     * Example 2: Using custom configuration
     */
    public static void customConfigExample() {
        logger.info("=== Custom Configuration Example ===");
        
        try {
            // Use predefined HSD1 configuration
            ModelConfig.TrainingConfig trainConfig = ModelConfig.getHSD1Config();
            ModelConfig.DataConfig dataConfig = ModelConfig.getDefaultDataConfig();
            
            // Modify some parameters
            dataConfig.setNumStudents(300);
            dataConfig.setQAposValue(3);
            trainConfig.setEpochs(75);
            
            // Setup with custom config
            INDArray qMatrix = DataProcessor.getSampleQMatrix1();
            INDArray qApos = DataProcessor.generateQApos(
                dataConfig.getQAposValue(), (int) qMatrix.shape()[1]);
            INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
            
            QNet qnet = new QNet(qMatrix, qStar);
            qnet.initializeNetwork((int) qMatrix.shape()[0], (int) qMatrix.shape()[1]);
            
            // Generate data with custom size
            DataProcessor.DataSplit data = DataProcessor.generateSyntheticData(
                qMatrix, dataConfig.getNumStudents(), trainConfig.getSeed());
            
            DataSetIterator trainIter = DataProcessor.createDataSetIterator(
                data.getTrainFeatures(), data.getTrainLabels(), 
                trainConfig.getBatchSize(), trainConfig.isShuffle());
            DataSetIterator testIter = DataProcessor.createDataSetIterator(
                data.getTestFeatures(), data.getTestLabels(), 
                trainConfig.getBatchSize(), false);
            
            // Train with custom configuration
            TrainingHistory history = qnet.trainNetwork(
                trainIter, testIter, trainConfig.getEpochs(), trainConfig.isVerbose());
            
            System.out.println("Custom Configuration Results:");
            history.printSummary();
            
        } catch (Exception e) {
            logger.error("Error in custom config example: ", e);
        }
    }
    
    /**
     * Example 3: Testing with different Q-matrix configurations
     */
    public static void differentQMatrixExample() {
        logger.info("=== Different Q-Matrix Example ===");
        
        try {
            // Create a larger Q-matrix (15 items, 4 attributes)
            double[][] largerQData = {
                {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
                {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}, {0, 1, 1, 0},
                {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 0}, {1, 1, 0, 1},
                {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}
            };
            
            INDArray largerQMatrix = org.nd4j.linalg.factory.Nd4j.create(largerQData);
            INDArray qApos = DataProcessor.generateQApos(3, (int) largerQMatrix.shape()[1]);
            INDArray qStar = DataProcessor.generateQStar(largerQMatrix, qApos);
            
            System.out.println("Larger Q-matrix shape: " + java.util.Arrays.toString(largerQMatrix.shape()));
            System.out.println("Q-star shape: " + java.util.Arrays.toString(qStar.shape()));
            
            QNet qnet = new QNet(largerQMatrix, qStar);
            qnet.initializeNetwork((int) largerQMatrix.shape()[0], (int) largerQMatrix.shape()[1]);
            qnet.printArchitecture();
            
            // Generate more data for larger problem
            DataProcessor.DataSplit data = DataProcessor.generateSyntheticData(largerQMatrix, 800, 456L);
            
            DataSetIterator trainIter = DataProcessor.createDataSetIterator(
                data.getTrainFeatures(), data.getTrainLabels(), 64, true);
            DataSetIterator testIter = DataProcessor.createDataSetIterator(
                data.getTestFeatures(), data.getTestLabels(), 64, false);
            
            // Train with more epochs for larger problem
            TrainingHistory history = qnet.trainNetwork(trainIter, testIter, 120, true);
            
            System.out.println("Larger Q-Matrix Results:");
            history.printSummary();
            
        } catch (Exception e) {
            logger.error("Error in different Q-matrix example: ", e);
        }
    }
}
