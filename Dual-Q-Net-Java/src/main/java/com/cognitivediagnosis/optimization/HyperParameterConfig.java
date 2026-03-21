package com.cognitivediagnosis.optimization;

import java.util.Map;

/**
 * 超参数配置类
 * 封装了Dual Q-Net的所有可调超参数
 */
public class HyperParameterConfig {
    
    // 学习率
    public final double learningRate;
    
    // L2正则化系数
    public final double l2Regularization;
    
    // 偏置初始化值
    public final double biasInit;
    
    // 梯度裁剪阈值
    public final double gradientClippingThreshold;
    
    // 批处理大小
    public final int batchSize;
    
    // 训练轮数
    public final int epochs;
    
    // Dropout概率
    public final double dropoutRate;
    
    // 早停patience
    public final int earlyStoppingPatience;
    
    // 学习率衰减因子
    public final double learningRateDecay;
    
    // 约束应用强度
    public final double constraintStrength;
    
    public HyperParameterConfig(Map<String, Double> parameters) {
        this.learningRate = parameters.get("learningRate");
        this.l2Regularization = parameters.get("l2Regularization");
        this.biasInit = parameters.get("biasInit");
        this.gradientClippingThreshold = parameters.get("gradientClippingThreshold");
        this.batchSize = parameters.get("batchSize").intValue();
        this.epochs = parameters.get("epochs").intValue();
        this.dropoutRate = parameters.get("dropoutRate");
        this.earlyStoppingPatience = parameters.get("earlyStoppingPatience").intValue();
        this.learningRateDecay = parameters.get("learningRateDecay");
        this.constraintStrength = parameters.get("constraintStrength");
    }
    
    public HyperParameterConfig(double learningRate, double l2Regularization, double biasInit,
                               double gradientClippingThreshold, int batchSize, int epochs,
                               double dropoutRate, int earlyStoppingPatience, 
                               double learningRateDecay, double constraintStrength) {
        this.learningRate = learningRate;
        this.l2Regularization = l2Regularization;
        this.biasInit = biasInit;
        this.gradientClippingThreshold = gradientClippingThreshold;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.dropoutRate = dropoutRate;
        this.earlyStoppingPatience = earlyStoppingPatience;
        this.learningRateDecay = learningRateDecay;
        this.constraintStrength = constraintStrength;
    }
    
    @Override
    public String toString() {
        return String.format(
            "HyperParameterConfig{lr=%.4f, l2=%.6f, bias=%.3f, gradClip=%.1f, " +
            "batch=%d, epochs=%d, dropout=%.3f, patience=%d, lrDecay=%.4f, constraint=%.3f}",
            learningRate, l2Regularization, biasInit, gradientClippingThreshold,
            batchSize, epochs, dropoutRate, earlyStoppingPatience, 
            learningRateDecay, constraintStrength
        );
    }
}



