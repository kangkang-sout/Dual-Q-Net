package com.cognitivediagnosis.config;

/**
 * Configuration class for model parameters
 * Corresponds to configs.py in the Python implementation
 */
public class ModelConfig {
    
    // Training parameters
    public static class TrainingConfig {
        private double learningRate = 0.001;
        private int epochs = 100;
        private int batchSize = 64;
        private boolean shuffle = true;
        private boolean verbose = true;
        private long seed = 123L;
        
        // Getters and setters
        public double getLearningRate() { return learningRate; }
        public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
        
        public int getEpochs() { return epochs; }
        public void setEpochs(int epochs) { this.epochs = epochs; }
        
        public int getBatchSize() { return batchSize; }
        public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
        
        public boolean isShuffle() { return shuffle; }
        public void setShuffle(boolean shuffle) { this.shuffle = shuffle; }
        
        public boolean isVerbose() { return verbose; }
        public void setVerbose(boolean verbose) { this.verbose = verbose; }
        
        public long getSeed() { return seed; }
        public void setSeed(long seed) { this.seed = seed; }
    }
    
    // Data parameters
    public static class DataConfig {
        private int numStudents = 500;
        private int testSize = 100;
        private double testRatio = 0.2;
        private int qAposValue = 4;
        
        public int getNumStudents() { return numStudents; }
        public void setNumStudents(int numStudents) { this.numStudents = numStudents; }
        
        public int getTestSize() { return testSize; }
        public void setTestSize(int testSize) { this.testSize = testSize; }
        
        public double getTestRatio() { return testRatio; }
        public void setTestRatio(double testRatio) { this.testRatio = testRatio; }
        
        public int getQAposValue() { return qAposValue; }
        public void setQAposValue(int qAposValue) { this.qAposValue = qAposValue; }
    }
    
    // Predefined configurations for different datasets
    public static TrainingConfig getHSD1Config() {
        TrainingConfig config = new TrainingConfig();
        config.setLearningRate(0.008);
        config.setEpochs(100);
        config.setBatchSize(64);
        return config;
    }
    
    public static TrainingConfig getLSD1Config() {
        TrainingConfig config = new TrainingConfig();
        config.setLearningRate(0.005);
        config.setEpochs(200);
        config.setBatchSize(128);
        return config;
    }
    
    public static TrainingConfig getEDMConfig() {
        TrainingConfig config = new TrainingConfig();
        config.setLearningRate(0.004);
        config.setEpochs(100);
        config.setBatchSize(64);
        return config;
    }
    
    public static DataConfig getDefaultDataConfig() {
        return new DataConfig();
    }
}
