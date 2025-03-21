#ifndef SIMPLEFLOW_METRIC_H
#define SIMPLEFLOW_METRIC_H

#include "types.h"
#include <memory>

namespace simpleflow {

// 评估指标基类
class Metric {
public:
    virtual ~Metric() = default;
    
    // 添加一个样本的预测和真实值
    virtual void Add(Float prediction, Float target) = 0;
    
    // 添加一批样本的预测和真实值
    virtual void Add(const FloatVector& predictions, const FloatVector& targets);
    
    // 获取当前指标值
    virtual Float Get() const = 0;
    
    // 重置指标
    virtual void Reset() = 0;
    
    // 获取指标名称
    virtual String GetName() const = 0;
};

// 准确率指标
class AccuracyMetric : public Metric {
public:
    AccuracyMetric(Float threshold = 0.5);
    
    void Add(Float prediction, Float target) override;
    Float Get() const override;
    void Reset() override;
    String GetName() const override { return "Accuracy"; }
    
private:
    Int correct_count_;
    Int total_count_;
    Float threshold_;
};

// AUC指标
class AUCMetric : public Metric {
public:
    AUCMetric();
    
    void Add(Float prediction, Float target) override;
    void Add(const FloatVector& predictions, const FloatVector& targets) override;
    Float Get() const override;
    void Reset() override;
    String GetName() const override { return "AUC"; }
    
private:
    std::vector<std::pair<Float, Int>> predictions_; // (预测值, 真实标签)
    Float CalculateAUC() const;
};

// 对数损失指标
class LogLossMetric : public Metric {
public:
    LogLossMetric();
    
    void Add(Float prediction, Float target) override;
    Float Get() const override;
    void Reset() override;
    String GetName() const override { return "LogLoss"; }
    
private:
    Float sum_loss_;
    Int count_;
};

} // namespace simpleflow

#endif // SIMPLEFLOW_METRIC_H 