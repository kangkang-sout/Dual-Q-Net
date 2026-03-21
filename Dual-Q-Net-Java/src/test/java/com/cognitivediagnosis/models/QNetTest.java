package com.cognitivediagnosis.models;

import com.cognitivediagnosis.data.DataProcessor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.*;

/**
 * Test class for QNet model
 */
public class QNetTest {
    
    @Test
    public void testQNetInitialization() {
        // Load sample Q-matrix
        INDArray qMatrix = DataProcessor.getSampleQMatrix1();
        
        // Generate Q-star matrix
        INDArray qApos = DataProcessor.generateQApos(2, (int) qMatrix.shape()[1]);
        INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
        
        // Create QNet
        QNet qnet = new QNet(qMatrix, qStar);
        
        // Initialize network
        int inputSize = (int) qMatrix.shape()[0];
        int outputSize = (int) qMatrix.shape()[1];
        qnet.initializeNetwork(inputSize, outputSize);
        
        // Verify network is initialized
        assertNotNull(qnet.getNetwork());
        assertEquals("Dual Q-Net", qnet.getModelName());
        
        // Verify matrices
        assertArrayEquals(qMatrix.shape(), qnet.getQMatrix().shape());
        assertArrayEquals(qStar.shape(), qnet.getQStarMatrix().shape());
    }
    
    @Test
    public void testQNetForwardPass() {
        // Setup
        INDArray qMatrix = DataProcessor.getSampleQMatrix1();
        INDArray qApos = DataProcessor.generateQApos(2, (int) qMatrix.shape()[1]);
        INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
        
        QNet qnet = new QNet(qMatrix, qStar);
        qnet.initializeNetwork((int) qMatrix.shape()[0], (int) qMatrix.shape()[1]);
        
        // Generate test data
        DataProcessor.DataSplit data = DataProcessor.generateSyntheticData(qMatrix, 10, 123L);
        
        // Forward pass
        INDArray predictions = qnet.forward(data.getTestFeatures());
        
        // Verify output shape
        assertArrayEquals(new long[]{data.getTestFeatures().shape()[0], qMatrix.shape()[1]}, 
                         predictions.shape());
        
        // Verify output range (should be between 0 and 1 due to sigmoid)
        assertTrue(predictions.minNumber().doubleValue() >= 0.0);
        assertTrue(predictions.maxNumber().doubleValue() <= 1.0);
    }
    
    @Test
    public void testMetricsCalculation() {
        // Create test data
        INDArray predictions = DataProcessor.getSampleQMatrix1().getRows(0, 5); // First 5 rows
        INDArray targets = DataProcessor.getSampleQMatrix1().getRows(0, 5);     // Same as targets (perfect match)
        
        // Calculate metrics
        double aar = BaseNet.calculateAAR(predictions, targets);
        double par = BaseNet.calculatePAR(predictions, targets);
        
        // Should be perfect scores
        assertEquals(1.0, aar, 0.01);
        assertEquals(1.0, par, 0.01);
    }
    
    @Test
    public void testDataProcessorQStarGeneration() {
        INDArray qMatrix = DataProcessor.getSampleQMatrix1();
        INDArray qApos = DataProcessor.generateQApos(2, (int) qMatrix.shape()[1]);
        INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);
        
        // Verify Q-star dimensions
        assertEquals(qMatrix.shape()[0], qStar.shape()[0]); // Same number of items
        assertEquals(qApos.shape()[0], qStar.shape()[1]);   // Number of q-apos patterns
        
        // Verify Q-star contains only 0s and 1s
        for (int i = 0; i < qStar.shape()[0]; i++) {
            for (int j = 0; j < qStar.shape()[1]; j++) {
                double value = qStar.getDouble(i, j);
                assertTrue(value == 0.0 || value == 1.0);
            }
        }
    }
}
