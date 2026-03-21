package com.cognitivediagnosis.utils;

import com.cognitivediagnosis.models.BaseNet;
import com.cognitivediagnosis.models.TrainingHistory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for comparing model performances
 */
public class ModelComparator {
    
    public static class ModelResult {
        private final String modelName;
        private final double finalTrainAAR;
        private final double finalTrainPAR;
        private final double finalTestAAR;
        private final double finalTestPAR;
        private final double finalTrainLoss;
        private final double finalTestLoss;
        
        public ModelResult(String modelName, TrainingHistory history) {
            this.modelName = modelName;
            
            List<Double> trainAAR = history.getTrainAAR();
            List<Double> trainPAR = history.getTrainPAR();
            List<Double> trainLosses = history.getTrainLosses();
            List<Double> testAAR = history.getTestAAR();
            List<Double> testPAR = history.getTestPAR();
            List<Double> testLosses = history.getTestLosses();
            
            this.finalTrainAAR = trainAAR.get(trainAAR.size() - 1);
            this.finalTrainPAR = trainPAR.get(trainPAR.size() - 1);
            this.finalTrainLoss = trainLosses.get(trainLosses.size() - 1);
            
            if (!testAAR.isEmpty()) {
                this.finalTestAAR = testAAR.get(testAAR.size() - 1);
                this.finalTestPAR = testPAR.get(testPAR.size() - 1);
                this.finalTestLoss = testLosses.get(testLosses.size() - 1);
            } else {
                this.finalTestAAR = 0.0;
                this.finalTestPAR = 0.0;
                this.finalTestLoss = 0.0;
            }
        }
        
        // Getters
        public String getModelName() { return modelName; }
        public double getFinalTrainAAR() { return finalTrainAAR; }
        public double getFinalTrainPAR() { return finalTrainPAR; }
        public double getFinalTestAAR() { return finalTestAAR; }
        public double getFinalTestPAR() { return finalTestPAR; }
        public double getFinalTrainLoss() { return finalTrainLoss; }
        public double getFinalTestLoss() { return finalTestLoss; }
    }
    
    /**
     * Compare multiple models and print results
     */
    public static void compareModels(List<ModelResult> results) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("MODEL COMPARISON RESULTS");
        System.out.println("=".repeat(80));
        
        // Header
        System.out.printf("%-12s %-10s %-10s %-10s %-10s %-10s %-10s%n",
            "Model", "Train AAR", "Train PAR", "Test AAR", "Test PAR", "Train Loss", "Test Loss");
        System.out.println("-".repeat(80));
        
        // Results
        for (ModelResult result : results) {
            System.out.printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n",
                result.getModelName(),
                result.getFinalTrainAAR(),
                result.getFinalTrainPAR(),
                result.getFinalTestAAR(),
                result.getFinalTestPAR(),
                result.getFinalTrainLoss(),
                result.getFinalTestLoss());
        }
        
        System.out.println("=".repeat(80));
        
        // Find best performers
        ModelResult bestTestAAR = results.stream()
            .max((r1, r2) -> Double.compare(r1.getFinalTestAAR(), r2.getFinalTestAAR()))
            .orElse(null);
        
        ModelResult bestTestPAR = results.stream()
            .max((r1, r2) -> Double.compare(r1.getFinalTestPAR(), r2.getFinalTestPAR()))
            .orElse(null);
        
        if (bestTestAAR != null) {
            System.out.printf("Best Test AAR: %s (%.4f)%n", 
                bestTestAAR.getModelName(), bestTestAAR.getFinalTestAAR());
        }
        
        if (bestTestPAR != null) {
            System.out.printf("Best Test PAR: %s (%.4f)%n", 
                bestTestPAR.getModelName(), bestTestPAR.getFinalTestPAR());
        }
    }
    
    /**
     * Calculate improvement percentage
     */
    public static double calculateImprovement(double baseline, double improved) {
        return ((improved - baseline) / baseline) * 100.0;
    }
    
    /**
     * Perform statistical significance test (simplified)
     */
    public static boolean isSignificantlyBetter(INDArray predictions1, INDArray predictions2, 
                                              INDArray targets, double threshold) {
        double aar1 = BaseNet.calculateAAR(predictions1, targets);
        double aar2 = BaseNet.calculateAAR(predictions2, targets);
        
        return Math.abs(aar2 - aar1) > threshold;
    }
    
    /**
     * Generate performance report
     */
    public static void generateReport(List<ModelResult> results, String baselineModel) {
        ModelResult baseline = results.stream()
            .filter(r -> r.getModelName().equals(baselineModel))
            .findFirst()
            .orElse(null);
        
        if (baseline == null) {
            System.out.println("Baseline model not found: " + baselineModel);
            return;
        }
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("PERFORMANCE IMPROVEMENT REPORT");
        System.out.println("Baseline: " + baselineModel);
        System.out.println("=".repeat(60));
        
        for (ModelResult result : results) {
            if (!result.getModelName().equals(baselineModel)) {
                double aarImprovement = calculateImprovement(
                    baseline.getFinalTestAAR(), result.getFinalTestAAR());
                double parImprovement = calculateImprovement(
                    baseline.getFinalTestPAR(), result.getFinalTestPAR());
                
                System.out.printf("%s vs %s:%n", result.getModelName(), baselineModel);
                System.out.printf("  AAR Improvement: %+.2f%%%n", aarImprovement);
                System.out.printf("  PAR Improvement: %+.2f%%%n", parImprovement);
                System.out.println();
            }
        }
    }
}
