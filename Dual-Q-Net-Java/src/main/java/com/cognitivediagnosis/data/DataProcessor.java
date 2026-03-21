package com.cognitivediagnosis.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Data processing utilities for cognitive diagnosis
 * Corresponds to data processing functions in Python implementation
 */
public class DataProcessor {
    
    private static final Logger logger = LoggerFactory.getLogger(DataProcessor.class);
    
    /**
     * Generate Q-star matrix from Q-matrix and q_apos
     * Corresponds to get_Qstar function in Python utils
     */
    public static INDArray generateQStar(INDArray qMatrix, INDArray qApos) {
        int numItems = (int) qMatrix.shape()[0];
        int numQStarAttributes = (int) qApos.shape()[0];
        
        INDArray qStar = Nd4j.zeros(numItems, numQStarAttributes);
        
        // For each item, check if it matches any q_apos pattern
        for (int i = 0; i < numItems; i++) {
            INDArray itemPattern = qMatrix.getRow(i);
            for (int j = 0; j < numQStarAttributes; j++) {
                INDArray qAposPattern = qApos.getRow(j);
                
                // Check if item pattern contains the q_apos pattern
                boolean matches = true;
                for (int k = 0; k < qAposPattern.length(); k++) {
                    if (qAposPattern.getDouble(k) == 1 && itemPattern.getDouble(k) == 0) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    qStar.putScalar(i, j, 1.0);
                }
            }
        }
        
        return qStar;
    }
    
    /**
     * Generate default q_apos matrix
     * Corresponds to get_q_apos function in Python utils
     */
    public static INDArray generateQApos(int qAposValue, int numAttributes) {
        if (qAposValue >= numAttributes) {
            // Generate all possible combinations
            List<INDArray> combinations = new ArrayList<>();
            generateCombinations(numAttributes, qAposValue, 0, new boolean[numAttributes], combinations);
            
            INDArray qApos = Nd4j.zeros(combinations.size(), numAttributes);
            for (int i = 0; i < combinations.size(); i++) {
                qApos.putRow(i, combinations.get(i));
            }
            return qApos;
        } else {
            // Generate combinations of size qAposValue
            List<INDArray> combinations = new ArrayList<>();
            generateFixedSizeCombinations(numAttributes, qAposValue, 0, new boolean[numAttributes], combinations, 0);
            
            INDArray qApos = Nd4j.zeros(combinations.size(), numAttributes);
            for (int i = 0; i < combinations.size(); i++) {
                qApos.putRow(i, combinations.get(i));
            }
            return qApos;
        }
    }
    
    private static void generateCombinations(int numAttributes, int minSize, int start, 
                                           boolean[] current, List<INDArray> combinations) {
        int currentCount = 0;
        for (boolean b : current) {
            if (b) currentCount++;
        }
        
        if (currentCount >= minSize) {
            INDArray combination = Nd4j.zeros(1, numAttributes);
            for (int i = 0; i < numAttributes; i++) {
                combination.putScalar(0, i, current[i] ? 1.0 : 0.0);
            }
            combinations.add(combination);
        }
        
        for (int i = start; i < numAttributes; i++) {
            current[i] = true;
            generateCombinations(numAttributes, minSize, i + 1, current, combinations);
            current[i] = false;
        }
    }
    
    private static void generateFixedSizeCombinations(int numAttributes, int size, int start,
                                                    boolean[] current, List<INDArray> combinations, int currentSize) {
        if (currentSize == size) {
            INDArray combination = Nd4j.zeros(1, numAttributes);
            for (int i = 0; i < numAttributes; i++) {
                combination.putScalar(0, i, current[i] ? 1.0 : 0.0);
            }
            combinations.add(combination);
            return;
        }
        
        for (int i = start; i < numAttributes; i++) {
            current[i] = true;
            generateFixedSizeCombinations(numAttributes, size, i + 1, current, combinations, currentSize + 1);
            current[i] = false;
        }
    }
    
    /**
     * Split data into train and test sets
     */
    public static DataSplit trainTestSplit(INDArray features, INDArray labels, double testRatio, long seed) {
        int numSamples = (int) features.shape()[0];
        int testSize = (int) (numSamples * testRatio);
        int trainSize = numSamples - testSize;
        
        // Create indices and shuffle
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(seed));
        
        // Split indices
        List<Integer> trainIndices = indices.subList(0, trainSize);
        List<Integer> testIndices = indices.subList(trainSize, numSamples);
        
        // Create train and test arrays
        INDArray trainFeatures = Nd4j.zeros(trainSize, features.shape()[1]);
        INDArray trainLabels = Nd4j.zeros(trainSize, labels.shape()[1]);
        INDArray testFeatures = Nd4j.zeros(testSize, features.shape()[1]);
        INDArray testLabels = Nd4j.zeros(testSize, labels.shape()[1]);
        
        for (int i = 0; i < trainSize; i++) {
            int idx = trainIndices.get(i);
            trainFeatures.putRow(i, features.getRow(idx));
            trainLabels.putRow(i, labels.getRow(idx));
        }
        
        for (int i = 0; i < testSize; i++) {
            int idx = testIndices.get(i);
            testFeatures.putRow(i, features.getRow(idx));
            testLabels.putRow(i, labels.getRow(idx));
        }
        
        return new DataSplit(trainFeatures, trainLabels, testFeatures, testLabels);
    }
    
    /**
     * Create DataSetIterator from features and labels
     */
    public static DataSetIterator createDataSetIterator(INDArray features, INDArray labels, 
                                                       int batchSize, boolean shuffle) {
        DataSet dataSet = new DataSet(features, labels);
        List<DataSet> dataSetList = dataSet.asList();
        
        if (shuffle) {
            Collections.shuffle(dataSetList);
        }
        
        return new SimpleDataSetIterator(dataSetList, batchSize);
    }
    
    /**
     * Load sample Q-matrix (corresponds to q1 in Python data.py)
     */
    public static INDArray getSampleQMatrix1() {
        double[][] qData = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 0},
            {1, 0, 1},
            {1, 1, 0},
            {0, 1, 1},
            {1, 1, 1}
        };
        return Nd4j.create(qData);
    }
    
    /**
     * Generate synthetic data for testing
     */
    public static DataSplit generateSyntheticData(INDArray qMatrix, int numStudents, long seed) {
        Random random = new Random(seed);
        int numItems = (int) qMatrix.shape()[0];
        int numAttributes = (int) qMatrix.shape()[1];
        
        // Generate student attribute patterns
        INDArray studentAttributes = Nd4j.zeros(numStudents, numAttributes);
        for (int i = 0; i < numStudents; i++) {
            for (int j = 0; j < numAttributes; j++) {
                studentAttributes.putScalar(i, j, random.nextDouble() > 0.5 ? 1.0 : 0.0);
            }
        }
        
        // Generate responses based on DINA model
        INDArray responses = Nd4j.zeros(numStudents, numItems);
        for (int i = 0; i < numStudents; i++) {
            for (int j = 0; j < numItems; j++) {
                // Check if student has all required attributes for item j
                boolean hasAllAttributes = true;
                for (int k = 0; k < numAttributes; k++) {
                    if (qMatrix.getDouble(j, k) == 1 && studentAttributes.getDouble(i, k) == 0) {
                        hasAllAttributes = false;
                        break;
                    }
                }
                
                // Generate response with some noise
                double correctProb = hasAllAttributes ? 0.9 : 0.1;
                responses.putScalar(i, j, random.nextDouble() < correctProb ? 1.0 : 0.0);
            }
        }
        
        return trainTestSplit(responses, studentAttributes, 0.2, seed);
    }
    
    /**
     * Data split container class
     */
    public static class DataSplit {
        private final INDArray trainFeatures;
        private final INDArray trainLabels;
        private final INDArray testFeatures;
        private final INDArray testLabels;
        
        public DataSplit(INDArray trainFeatures, INDArray trainLabels, 
                        INDArray testFeatures, INDArray testLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.testFeatures = testFeatures;
            this.testLabels = testLabels;
        }
        
        public INDArray getTrainFeatures() { return trainFeatures; }
        public INDArray getTrainLabels() { return trainLabels; }
        public INDArray getTestFeatures() { return testFeatures; }
        public INDArray getTestLabels() { return testLabels; }
    }
    
    /**
     * Simple DataSetIterator implementation
     */
    private static class SimpleDataSetIterator implements DataSetIterator {
        private final List<DataSet> dataSets;
        private final int batchSize;
        private int currentIndex = 0;
        
        public SimpleDataSetIterator(List<DataSet> dataSets, int batchSize) {
            this.dataSets = dataSets;
            this.batchSize = batchSize;
        }
        
        @Override
        public DataSet next(int num) {
            List<DataSet> batch = new ArrayList<>();
            for (int i = 0; i < num && hasNext(); i++) {
                batch.add(dataSets.get(currentIndex++));
            }
            
            if (batch.isEmpty()) {
                return null;
            }
            
            // Merge all datasets in batch
            List<INDArray> features = new ArrayList<>();
            List<INDArray> labels = new ArrayList<>();
            
            for (DataSet ds : batch) {
                features.add(ds.getFeatures());
                labels.add(ds.getLabels());
            }
            
            INDArray mergedFeatures = Nd4j.vstack(features.toArray(new INDArray[0]));
            INDArray mergedLabels = Nd4j.vstack(labels.toArray(new INDArray[0]));
            
            return new DataSet(mergedFeatures, mergedLabels);
        }
        
        @Override
        public int inputColumns() {
            return dataSets.isEmpty() ? 0 : (int) dataSets.get(0).getFeatures().size(1);
        }
        
        @Override
        public int totalOutcomes() {
            return dataSets.isEmpty() ? 0 : (int) dataSets.get(0).getLabels().size(1);
        }
        
        @Override
        public boolean resetSupported() {
            return true;
        }
        
        @Override
        public boolean asyncSupported() {
            return false;
        }
        
        @Override
        public void reset() {
            currentIndex = 0;
        }
        
        @Override
        public int batch() {
            return batchSize;
        }
        
        @Override
        public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
            // Not implemented
        }
        
        @Override
        public org.nd4j.linalg.dataset.api.DataSetPreProcessor getPreProcessor() {
            return null;
        }
        
        @Override
        public List<String> getLabels() {
            return null;
        }
        
        @Override
        public boolean hasNext() {
            return currentIndex < dataSets.size();
        }
        
        @Override
        public DataSet next() {
            return next(batchSize);
        }
    }
}
