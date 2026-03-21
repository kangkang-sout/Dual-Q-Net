package com.cognitivediagnosis.models;

import java.util.ArrayList;
import java.util.List;

/**
 * Class to store training history metrics
 * Corresponds to the dfhistory DataFrame in Python
 */
public class TrainingHistory {
    private final List<Double> trainLosses;
    private final List<Double> trainAAR;
    private final List<Double> trainPAR;
    private final List<Double> testLosses;
    private final List<Double> testAAR;
    private final List<Double> testPAR;
    
    // Default constructor for mutable history
    public TrainingHistory() {
        this.trainLosses = new ArrayList<>();
        this.trainAAR = new ArrayList<>();
        this.trainPAR = new ArrayList<>();
        this.testLosses = new ArrayList<>();
        this.testAAR = new ArrayList<>();
        this.testPAR = new ArrayList<>();
    }
    
    public TrainingHistory(List<Double> trainLosses, List<Double> trainAAR, List<Double> trainPAR,
                          List<Double> testLosses, List<Double> testAAR, List<Double> testPAR) {
        this.trainLosses = trainLosses;
        this.trainAAR = trainAAR;
        this.trainPAR = trainPAR;
        this.testLosses = testLosses;
        this.testAAR = testAAR;
        this.testPAR = testPAR;
    }
    
    public List<Double> getTrainLosses() { return trainLosses; }
    public List<Double> getTrainAAR() { return trainAAR; }
    public List<Double> getTrainPAR() { return trainPAR; }
    public List<Double> getTestLosses() { return testLosses; }
    public List<Double> getTestAAR() { return testAAR; }
    public List<Double> getTestPAR() { return testPAR; }
    
    /**
     * Add epoch metrics to history
     */
    public void addEpoch(double trainLoss, double trainAAR, double trainPAR, 
                        double testLoss, double testAAR, double testPAR) {
        this.trainLosses.add(trainLoss);
        this.trainAAR.add(trainAAR);
        this.trainPAR.add(trainPAR);
        this.testLosses.add(testLoss);
        this.testAAR.add(testAAR);
        this.testPAR.add(testPAR);
    }
    
    public void printSummary() {
        System.out.println("Training Summary:");
        System.out.println("Final Train Loss: " + trainLosses.get(trainLosses.size()-1));
        System.out.println("Final Train AAR: " + trainAAR.get(trainAAR.size()-1));
        System.out.println("Final Train PAR: " + trainPAR.get(trainPAR.size()-1));
        
        if (!testLosses.isEmpty()) {
            System.out.println("Final Test Loss: " + testLosses.get(testLosses.size()-1));
            System.out.println("Final Test AAR: " + testAAR.get(testAAR.size()-1));
            System.out.println("Final Test PAR: " + testPAR.get(testPAR.size()-1));
        }
    }
}
