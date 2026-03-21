package com.cognitivediagnosis;

import com.cognitivediagnosis.data.DataProcessor;
import com.cognitivediagnosis.models.ANN;
import com.cognitivediagnosis.models.BaseNet;
import com.cognitivediagnosis.models.MLP;
import com.cognitivediagnosis.models.QNet;
import com.cognitivediagnosis.models.TrainingHistory;
import com.cognitivediagnosis.utils.ModelComparator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Main class demonstrating Dual Q-Net usage
 * Corresponds to main.py in the Python implementation
 */
public class Main {
    
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    
    public static void main(String[] args) {
        logger.info("Starting Dual Q-Net Java Implementation");
        
        try {
            // Load sample Q-matrix
            INDArray qMatrix = DataProcessor.getSampleQMatrix1();
            logger.info("Q-matrix shape: " + java.util.Arrays.toString(qMatrix.shape()));
            
            // Generate Q-apos and Q-star matrices
            int qAposValue = 2;
            int numAttributes = (int) qMatrix.shape()[1];
            INDArray qApos = DataProcessor.generateQApos(qAposValue, numAttributes);
            INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
            
            logger.info("Q-apos shape: " + java.util.Arrays.toString(qApos.shape()));
            logger.info("Q-star shape: " + java.util.Arrays.toString(qStar.shape()));
            
            // Generate synthetic training data - 增加数据量以提升泛化能力
            int numStudents = 2000;  // Python版本通常使用更多数据
            DataProcessor.DataSplit dataSplit = DataProcessor.generateSyntheticData(qMatrix, numStudents, 123L);
            
            logger.info("Training data shape: " + java.util.Arrays.toString(dataSplit.getTrainFeatures().shape()));
            logger.info("Test data shape: " + java.util.Arrays.toString(dataSplit.getTestFeatures().shape()));
            
            // Create data iterators - 优化：使用更小的batch size提升训练精度
            int batchSize = 16;  // 从64减少到16，更精细的梯度更新
            DataSetIterator trainIterator = DataProcessor.createDataSetIterator(
                dataSplit.getTrainFeatures(), dataSplit.getTrainLabels(), batchSize, true);
            DataSetIterator testIterator = DataProcessor.createDataSetIterator(
                dataSplit.getTestFeatures(), dataSplit.getTestLabels(), batchSize, false);
            
            // Initialize models
            MLP mlp = new MLP();
            ANN ann = new ANN();
            QNet qnet = new QNet(qMatrix, qStar);
            
            int inputSize = (int) qMatrix.shape()[0];
            int outputSize = (int) qMatrix.shape()[1];
            
            mlp.initializeNetwork(inputSize, outputSize);
            ann.initializeNetwork(inputSize, outputSize);
            qnet.initializeNetwork(inputSize, outputSize);
            
            // Print QNet architecture
            qnet.printArchitecture();
            
            // Training parameters - 匹配Python版本
            int epochs = 100; // 减少轮数避免梯度爆炸
            boolean verbose = true;
            
            logger.info("Starting training...");
            
            // Train MLP
            logger.info("Training MLP...");
            TrainingHistory mlpHistory = mlp.trainNetwork(trainIterator, testIterator, epochs, verbose);
            
            // Reset iterators
            trainIterator.reset();
            testIterator.reset();
            
            // Train ANN
            logger.info("Training ANN...");
            TrainingHistory annHistory = ann.trainNetwork(trainIterator, testIterator, epochs, verbose);
            
            // Reset iterators
            trainIterator.reset();
            testIterator.reset();
            
            // Train QNet (暂时使用基础版本直到贝叶斯优化修复)
            logger.info("Training Dual Q-Net...");
            TrainingHistory qnetHistory = qnet.trainNetwork(trainIterator, testIterator, epochs, verbose);
            
            // Create model results for comparison
            List<ModelComparator.ModelResult> results = new ArrayList<>();
            results.add(new ModelComparator.ModelResult("MLP", mlpHistory));
            results.add(new ModelComparator.ModelResult("ANN", annHistory));
            results.add(new ModelComparator.ModelResult("Dual Q-Net", qnetHistory));
            
            // Compare models
            ModelComparator.compareModels(results);
            
            // Generate improvement report with MLP as baseline
            ModelComparator.generateReport(results, "MLP");
            
            // Make predictions on test set for detailed analysis
            logger.info("\nMaking predictions on test set...");
            
            INDArray mlpPredictions = mlp.predict(dataSplit.getTestFeatures(), true);
            INDArray annPredictions = ann.predict(dataSplit.getTestFeatures(), true);
            INDArray qnetPredictions = qnet.predict(dataSplit.getTestFeatures(), true);
            
            // Calculate final metrics
            double mlpAAR = BaseNet.calculateAAR(mlpPredictions, dataSplit.getTestLabels());
            double mlpPAR = BaseNet.calculatePAR(mlpPredictions, dataSplit.getTestLabels());
            
            double annAAR = BaseNet.calculateAAR(annPredictions, dataSplit.getTestLabels());
            double annPAR = BaseNet.calculatePAR(annPredictions, dataSplit.getTestLabels());
            
            double qnetAAR = BaseNet.calculateAAR(qnetPredictions, dataSplit.getTestLabels());
            double qnetPAR = BaseNet.calculatePAR(qnetPredictions, dataSplit.getTestLabels());
            
            System.out.println("\n" + "=".repeat(50));
            System.out.println("FINAL TEST METRICS (Binary Predictions)");
            System.out.println("=".repeat(50));
            System.out.printf("MLP  - AAR: %.4f, PAR(%d): %.4f%n", mlpAAR, numAttributes, mlpPAR);
            System.out.printf("ANN  - AAR: %.4f, PAR(%d): %.4f%n", annAAR, numAttributes, annPAR);
            System.out.printf("QNet - AAR: %.4f, PAR(%d): %.4f%n", qnetAAR, numAttributes, qnetPAR);
            
            // Check for significant improvements
            boolean qnetVsMlp = ModelComparator.isSignificantlyBetter(
                mlpPredictions, qnetPredictions, dataSplit.getTestLabels(), 0.05);
            boolean qnetVsAnn = ModelComparator.isSignificantlyBetter(
                annPredictions, qnetPredictions, dataSplit.getTestLabels(), 0.05);
            
            System.out.println("\n" + "=".repeat(50));
            System.out.println("STATISTICAL SIGNIFICANCE (threshold: 5%)");
            System.out.println("=".repeat(50));
            System.out.println("QNet vs MLP: " + (qnetVsMlp ? "Significantly Better" : "Not Significant"));
            System.out.println("QNet vs ANN: " + (qnetVsAnn ? "Significantly Better" : "Not Significant"));
            
            logger.info("Training completed successfully!");
            
        } catch (Exception e) {
            logger.error("Error during execution: ", e);
            System.exit(1);
        }
    }
}
