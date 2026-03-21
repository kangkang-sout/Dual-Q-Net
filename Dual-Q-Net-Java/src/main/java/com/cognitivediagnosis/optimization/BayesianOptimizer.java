package com.cognitivediagnosis.optimization;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * 简化的贝叶斯优化器用于超参数调优
 * 使用高斯过程代理模型和采集函数进行高效搜索
 */
public class BayesianOptimizer {
    
    private static final Logger logger = LoggerFactory.getLogger(BayesianOptimizer.class);
    
    private final List<HyperParameter> hyperParameters;
    private final List<Observation> observations;
    private final Random random;
    private final int maxIterations;
    private final int nRandomStarts;
    
    public static class HyperParameter {
        public final String name;
        public final double minValue;
        public final double maxValue;
        public final boolean isInteger;
        
        public HyperParameter(String name, double minValue, double maxValue, boolean isInteger) {
            this.name = name;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.isInteger = isInteger;
        }
        
        public double sample() {
            double value = ThreadLocalRandom.current().nextDouble(minValue, maxValue);
            return isInteger ? Math.round(value) : value;
        }
        
        public double normalize(double value) {
            return (value - minValue) / (maxValue - minValue);
        }
        
        public double denormalize(double normalizedValue) {
            double value = normalizedValue * (maxValue - minValue) + minValue;
            return isInteger ? Math.round(value) : value;
        }
    }
    
    public static class Observation {
        public final Map<String, Double> parameters;
        public final double objective;
        
        public Observation(Map<String, Double> parameters, double objective) {
            this.parameters = new HashMap<>(parameters);
            this.objective = objective;
        }
    }
    
    public interface ObjectiveFunction {
        double evaluate(Map<String, Double> parameters);
    }
    
    public BayesianOptimizer(List<HyperParameter> hyperParameters, int maxIterations, int nRandomStarts) {
        this.hyperParameters = hyperParameters;
        this.observations = new ArrayList<>();
        this.random = new Random(42);
        this.maxIterations = maxIterations;
        this.nRandomStarts = nRandomStarts;
    }
    
    /**
     * 执行贝叶斯优化
     */
    public Map<String, Double> optimize(ObjectiveFunction objectiveFunction) {
        logger.info("开始贝叶斯优化，最大迭代次数: {}, 随机初始点: {}", maxIterations, nRandomStarts);
        
        // 第一阶段：随机采样初始点
        for (int i = 0; i < nRandomStarts; i++) {
            Map<String, Double> params = sampleRandomParameters();
            double objective = objectiveFunction.evaluate(params);
            observations.add(new Observation(params, objective));
            
            logger.info("随机初始化 {}/{}: AAR = {:.4f}, 参数: {}", 
                i + 1, nRandomStarts, objective, formatParameters(params));
        }
        
        // 第二阶段：贝叶斯优化迭代
        for (int i = nRandomStarts; i < maxIterations; i++) {
            Map<String, Double> nextParams = acquireNext();
            double objective = objectiveFunction.evaluate(nextParams);
            observations.add(new Observation(nextParams, objective));
            
            logger.info("贝叶斯优化 {}/{}: AAR = {:.4f}, 参数: {}", 
                i + 1, maxIterations, objective, formatParameters(nextParams));
        }
        
        // 返回最佳参数
        Observation best = getBestObservation();
        logger.info("贝叶斯优化完成！最佳AAR: {:.4f}, 最佳参数: {}", 
            best.objective, formatParameters(best.parameters));
        
        return best.parameters;
    }
    
    /**
     * 随机采样参数
     */
    private Map<String, Double> sampleRandomParameters() {
        Map<String, Double> params = new HashMap<>();
        for (HyperParameter hp : hyperParameters) {
            params.put(hp.name, hp.sample());
        }
        return params;
    }
    
    /**
     * 使用采集函数获取下一个评估点
     * 这里使用简化的Upper Confidence Bound (UCB)策略
     */
    private Map<String, Double> acquireNext() {
        int nCandidates = 1000;
        Map<String, Double> bestCandidate = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < nCandidates; i++) {
            Map<String, Double> candidate = sampleRandomParameters();
            double score = calculateAcquisitionScore(candidate);
            
            if (score > bestScore) {
                bestScore = score;
                bestCandidate = candidate;
            }
        }
        
        return bestCandidate;
    }
    
    /**
     * 计算采集分数 (简化的UCB)
     * UCB = 预测均值 + β * 预测标准差
     */
    private double calculateAcquisitionScore(Map<String, Double> candidate) {
        if (observations.isEmpty()) {
            return random.nextGaussian();
        }
        
        // 简化的高斯过程预测
        double[] predictions = predictGaussianProcess(candidate);
        double mean = predictions[0];
        double std = predictions[1];
        
        // UCB参数，平衡探索和开发
        double beta = 2.0;
        return mean + beta * std;
    }
    
    /**
     * 简化的高斯过程预测
     * 返回 [均值, 标准差]
     */
    private double[] predictGaussianProcess(Map<String, Double> candidate) {
        if (observations.isEmpty()) {
            return new double[]{0.0, 1.0};
        }
        
        // 计算与历史观察点的相似性权重
        double totalWeight = 0.0;
        double weightedSum = 0.0;
        double lengthScale = 0.3; // RBF核的长度尺度
        
        for (Observation obs : observations) {
            double distance = calculateDistance(candidate, obs.parameters);
            double weight = Math.exp(-distance * distance / (2 * lengthScale * lengthScale));
            
            totalWeight += weight;
            weightedSum += weight * obs.objective;
        }
        
        double mean = totalWeight > 0 ? weightedSum / totalWeight : 0.0;
        
        // 简化的不确定性估计
        double uncertainty = Math.exp(-totalWeight * 0.1);
        double std = Math.max(0.01, uncertainty * getObjectiveStd());
        
        return new double[]{mean, std};
    }
    
    /**
     * 计算参数间的标准化欧几里得距离
     */
    private double calculateDistance(Map<String, Double> params1, Map<String, Double> params2) {
        double sumSquaredDiff = 0.0;
        
        for (HyperParameter hp : hyperParameters) {
            double val1 = hp.normalize(params1.get(hp.name));
            double val2 = hp.normalize(params2.get(hp.name));
            sumSquaredDiff += (val1 - val2) * (val1 - val2);
        }
        
        return Math.sqrt(sumSquaredDiff);
    }
    
    /**
     * 获取目标函数的标准差
     */
    private double getObjectiveStd() {
        if (observations.size() < 2) {
            return 1.0;
        }
        
        double mean = observations.stream().mapToDouble(obs -> obs.objective).average().orElse(0.0);
        double variance = observations.stream()
            .mapToDouble(obs -> (obs.objective - mean) * (obs.objective - mean))
            .average().orElse(1.0);
        
        return Math.sqrt(variance);
    }
    
    /**
     * 获取最佳观察结果
     */
    public Observation getBestObservation() {
        return observations.stream()
            .max(Comparator.comparingDouble(obs -> obs.objective))
            .orElse(null);
    }
    
    /**
     * 格式化参数输出
     */
    private String formatParameters(Map<String, Double> params) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Double> entry : params.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            
            HyperParameter hp = hyperParameters.stream()
                .filter(h -> h.name.equals(entry.getKey()))
                .findFirst().orElse(null);
            
            if (hp != null && hp.isInteger) {
                sb.append(entry.getKey()).append("=").append(entry.getValue().intValue());
            } else {
                sb.append(entry.getKey()).append("=").append(String.format("%.4f", entry.getValue()));
            }
        }
        sb.append("}");
        return sb.toString();
    }
    
    /**
     * 获取所有观察结果
     */
    public List<Observation> getObservations() {
        return new ArrayList<>(observations);
    }
}
