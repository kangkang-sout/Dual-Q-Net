package com.cognitivediagnosis.optimization;

import com.cognitivediagnosis.data.DataProcessor;
import com.cognitivediagnosis.models.QNet;
import com.cognitivediagnosis.models.TrainingHistory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Dual Q-Net贝叶斯超参数优化器
 * 专门针对认知诊断任务优化Dual Q-Net性能
 */
public class DualQNetOptimizer {
    
    private static final Logger logger = LoggerFactory.getLogger(DualQNetOptimizer.class);
    
    private final INDArray qMatrix;
    private final INDArray qStarMatrix;
    private final DataProcessor.DataSplit dataSplit;
    private final BayesianOptimizer bayesianOptimizer;
    
    public DualQNetOptimizer(INDArray qMatrix, INDArray qStarMatrix, DataProcessor.DataSplit dataSplit) {
        this.qMatrix = qMatrix;
        this.qStarMatrix = qStarMatrix;
        this.dataSplit = dataSplit;
        
        // 定义超参数搜索空间
        List<BayesianOptimizer.HyperParameter> hyperParameters = createHyperParameterSpace();
        
        // 创建贝叶斯优化器
        this.bayesianOptimizer = new BayesianOptimizer(hyperParameters, 25, 8);
    }
    
    /**
     * 创建超参数搜索空间
     */
    private List<BayesianOptimizer.HyperParameter> createHyperParameterSpace() {
        List<BayesianOptimizer.HyperParameter> hyperParameters = new ArrayList<>();
        
        // 学习率：对数空间搜索
        hyperParameters.add(new BayesianOptimizer.HyperParameter("learningRate", 0.0001, 0.01, false));
        
        // L2正则化：对数空间
        hyperParameters.add(new BayesianOptimizer.HyperParameter("l2Regularization", 1e-6, 1e-3, false));
        
        // 偏置初始化
        hyperParameters.add(new BayesianOptimizer.HyperParameter("biasInit", 0.0, 0.1, false));
        
        // 梯度裁剪阈值
        hyperParameters.add(new BayesianOptimizer.HyperParameter("gradientClippingThreshold", 0.5, 2.0, false));
        
        // 批处理大小
        hyperParameters.add(new BayesianOptimizer.HyperParameter("batchSize", 8, 32, true));
        
        // 训练轮数
        hyperParameters.add(new BayesianOptimizer.HyperParameter("epochs", 200, 800, true));
        
        // Dropout概率
        hyperParameters.add(new BayesianOptimizer.HyperParameter("dropoutRate", 0.0, 0.3, false));
        
        // 早停patience
        hyperParameters.add(new BayesianOptimizer.HyperParameter("earlyStoppingPatience", 10, 50, true));
        
        // 学习率衰减
        hyperParameters.add(new BayesianOptimizer.HyperParameter("learningRateDecay", 0.95, 1.0, false));
        
        // 约束应用强度
        hyperParameters.add(new BayesianOptimizer.HyperParameter("constraintStrength", 0.5, 1.5, false));
        
        return hyperParameters;
    }
    
    /**
     * 执行贝叶斯优化
     */
    public HyperParameterConfig optimize() {
        logger.info("=== 开始Dual Q-Net贝叶斯超参数优化 ===");
        logger.info("搜索空间包含10个超参数，将执行25次评估");
        
        Map<String, Double> bestParams = bayesianOptimizer.optimize(this::evaluateHyperParameters);
        
        HyperParameterConfig bestConfig = new HyperParameterConfig(bestParams);
        logger.info("=== 贝叶斯优化完成 ===");
        logger.info("最佳超参数配置: {}", bestConfig);
        
        return bestConfig;
    }
    
    /**
     * 评估给定超参数配置的性能
     * 返回测试集AAR作为目标函数值
     */
    private double evaluateHyperParameters(Map<String, Double> parameters) {
        try {
            HyperParameterConfig config = new HyperParameterConfig(parameters);
            
            // 创建优化的QNet实例
            OptimizedQNet qnet = new OptimizedQNet(qMatrix, qStarMatrix, config);
            
            int inputSize = (int) qMatrix.shape()[0];
            int outputSize = (int) qMatrix.shape()[1];
            qnet.initializeNetwork(inputSize, outputSize);
            
            // 创建数据迭代器
            DataSetIterator trainIterator = DataProcessor.createDataSetIterator(
                dataSplit.getTrainFeatures(), dataSplit.getTrainLabels(), config.batchSize, true);
            DataSetIterator testIterator = DataProcessor.createDataSetIterator(
                dataSplit.getTestFeatures(), dataSplit.getTestLabels(), config.batchSize, false);
            
            // 训练模型
            TrainingHistory history = qnet.trainNetwork(trainIterator, testIterator, config.epochs, false);
            
            // 返回最佳测试AAR
            double bestTestAAR = history.getTestAAR().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            
            logger.debug("评估配置 {} -> AAR: {:.4f}", config, bestTestAAR);
            
            return bestTestAAR;
            
        } catch (Exception e) {
            logger.warn("超参数评估失败: {}", e.getMessage());
            return 0.0; // 返回最低分数表示失败的配置
        }
    }
    
    /**
     * 获取优化历史
     */
    public List<BayesianOptimizer.Observation> getOptimizationHistory() {
        return bayesianOptimizer.getObservations();
    }
    
    /**
     * 优化版本的QNet，支持动态超参数配置
     */
    public class OptimizedQNet extends QNet {
        
        private final HyperParameterConfig config;
        
        public OptimizedQNet(INDArray qMatrix, INDArray qStarMatrix, HyperParameterConfig config) {
            super(qMatrix, qStarMatrix);
            this.config = config;
        }
        
        @Override
        protected org.deeplearning4j.nn.conf.MultiLayerConfiguration buildNetworkConfiguration(int inputSize, int outputSize) {
            return new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER_UNIFORM)
                .updater(new org.nd4j.linalg.learning.config.Adam(config.learningRate))
                .l2(config.l2Regularization)
                .biasInit(config.biasInit)
                .gradientNormalization(org.deeplearning4j.nn.conf.GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(config.gradientClippingThreshold)
                .list()
                // Main constraint layer (mc): items -> attributes
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                    .nIn(inputSize)
                    .nOut(numAttributes)
                    .activation(org.nd4j.linalg.activations.Activation.RELU)
                    .dropOut(config.dropoutRate)
                    .build())
                // Latent constraint layer (lc): items -> q_star attributes  
                .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                    .nIn(inputSize)
                    .nOut(numQStarAttributes)
                    .activation(org.nd4j.linalg.activations.Activation.TANH)
                    .dropOut(config.dropoutRate)
                    .build())
                // Combination layer (cc): (attributes + q_star) -> attributes
                .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                        org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE)
                    .nIn(numAttributes + numQStarAttributes)
                    .nOut(outputSize)
                    .activation(org.nd4j.linalg.activations.Activation.SIGMOID)
                    .build())
                .build();
        }
        
        @Override
        public TrainingHistory trainNetwork(DataSetIterator trainIterator,
                                          DataSetIterator testIterator,
                                          int epochs,
                                          boolean verbose) {
            TrainingHistory history = new TrainingHistory();
            double bestTestAAR = 0.0;
            int patienceCounter = 0;
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                double epochLoss = 0.0;
                int batchCount = 0;
                trainIterator.reset();
                
                // 学习率衰减
                if (config.learningRateDecay < 1.0 && epoch > 0) {
                    double newLr = config.learningRate * Math.pow(config.learningRateDecay, epoch);
                    network.setLearningRate(newLr);
                }
                
                while (trainIterator.hasNext()) {
                    org.nd4j.linalg.dataset.DataSet batch = trainIterator.next();
                    INDArray predictions = network.output(batch.getFeatures());
                    double batchLoss = computeLoss(predictions, batch.getLabels());
                    epochLoss += batchLoss;
                    batchCount++;
                    network.fit(batch);
                    
                    // 应用约束，使用配置的强度
                    applyQMatrixConstraints(config.constraintStrength);
                    applyQStarConstraints(config.constraintStrength);
                }
                
                epochLoss /= batchCount;
                
                // 评估
                INDArray trainPreds = predict(getFullDataset(trainIterator), false);
                INDArray trainLabels = getFullLabels(trainIterator);
                double trainAAR = calculateAAR(trainPreds, trainLabels);
                double trainPAR = calculatePAR(trainPreds, trainLabels);
                
                double testLoss = 0.0, testAAR = 0.0, testPAR = 0.0;
                if (testIterator != null) {
                    INDArray testPreds = predict(getFullDataset(testIterator), false);
                    INDArray testLabels = getFullLabels(testIterator);
                    testLoss = computeLoss(testPreds, testLabels);
                    testAAR = calculateAAR(testPreds, testLabels);
                    testPAR = calculatePAR(testPreds, testLabels);
                }
                
                history.addEpoch(epochLoss, trainAAR, trainPAR, testLoss, testAAR, testPAR);
                
                // 早停检查
                if (testAAR > bestTestAAR) {
                    bestTestAAR = testAAR;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                    if (patienceCounter >= config.earlyStoppingPatience) {
                        if (verbose) {
                            logger.debug("Early stopping at epoch {}, best test AAR: {:.4f}", epoch, bestTestAAR);
                        }
                        break;
                    }
                }
                
                if (verbose && epoch % Math.max(1, epochs / 10) == 0) {
                    logger.debug("Epoch {}: Train AAR={:.4f}, Test AAR={:.4f}", epoch, trainAAR, testAAR);
                }
            }
            
            return history;
        }
        
        private void applyQMatrixConstraints(double strength) {
            if (network != null && network.getLayer(0) != null) {
                INDArray mcWeights = network.getLayer(0).getParam("W");
                if (mcWeights != null && qMatrix != null) {
                    INDArray constraints = qMatrix.transpose();
                    if (mcWeights.shape()[0] == constraints.shape()[0] && 
                        mcWeights.shape()[1] == constraints.shape()[1]) {
                        // 软约束：线性插值
                        mcWeights.muli(1.0 - strength).addi(constraints.mul(strength).muli(mcWeights));
                    }
                }
            }
        }
        
        private void applyQStarConstraints(double strength) {
            if (network != null && network.getLayer(1) != null) {
                INDArray lcWeights = network.getLayer(1).getParam("W");
                if (lcWeights != null && qStarMatrix != null) {
                    INDArray constraints = qStarMatrix.transpose();
                    if (lcWeights.shape()[0] == constraints.shape()[0] && 
                        lcWeights.shape()[1] == constraints.shape()[1]) {
                        // 软约束：线性插值
                        lcWeights.muli(1.0 - strength).addi(constraints.mul(strength).muli(lcWeights));
                    }
                }
            }
        }
    }
}
