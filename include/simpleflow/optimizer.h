#ifndef SIMPLEFLOW_OPTIMIZER_H
#define SIMPLEFLOW_OPTIMIZER_H

#include "simpleflow/types.h"
#include <memory>
#include <unordered_map>

namespace simpleflow {

// 优化器基类
class Optimizer {
public:
    Optimizer(Float learning_rate, Float l2_reg = 0.0);
    virtual ~Optimizer() = default;
    
    // 更新单个参数
    virtual void Update(Int index, Float gradient, Float& parameter) = 0;
    
    // 更新参数指针版本
    virtual void UpdateParameter(Float* parameter, Float gradient) {
        if (!parameter) return;
        Float reg_grad = gradient;
        if (l2_reg_ > 0) {
            reg_grad += l2_reg_ * (*parameter);
        }
        *parameter -= learning_rate_ * reg_grad;
    }
    
    // 更新参数向量 - 新增方法，用于兼容测试
    virtual void Update(FloatVector& parameters, const FloatVector& gradients);
    
    // 获取当前学习率
    Float GetLearningRate() const { return learning_rate_; }
    
    // 获取L2正则化系数
    Float GetL2Reg() const { return l2_reg_; }
    
    // 设置学习率
    void SetLearningRate(Float learning_rate) {
        if (learning_rate <= 0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        learning_rate_ = learning_rate;
    }
    
    // 工厂方法创建优化器
    static std::shared_ptr<Optimizer> Create(OptimizerType type, Float learning_rate, Float l2_reg = 0.0);
    
protected:
    Float learning_rate_;
    Float l2_reg_;
};

// 随机梯度下降优化器
class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(Float learning_rate, Float l2_reg = 0.0);
    void Update(Int index, Float gradient, Float& parameter) override;
};

// Adagrad优化器
class AdagradOptimizer : public Optimizer {
public:
    AdagradOptimizer(Float learning_rate, Float l2_reg = 0.0, Float epsilon = 1e-8);
    void Update(Int index, Float gradient, Float& parameter) override;
    
private:
    std::unordered_map<Int, Float> squared_gradients_;
    Float epsilon_;
};

// RMSProp优化器 - 新增优化器
class RMSPropOptimizer : public Optimizer {
public:
    RMSPropOptimizer(Float learning_rate, Float l2_reg = 0.0, Float decay_rate = 0.9, Float epsilon = 1e-8);
    void Update(Int index, Float gradient, Float& parameter) override;
    
private:
    std::unordered_map<Int, Float> cache_;
    Float decay_rate_;
    Float epsilon_;
};

// Adam优化器
class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(Float learning_rate, Float l2_reg = 0.0, Float beta1 = 0.9, Float beta2 = 0.999, Float epsilon = 1e-8);
    void Update(Int index, Float gradient, Float& parameter) override;
    
    // 重写向量版本的Update方法
    void Update(FloatVector& parameters, const FloatVector& gradients) override;
    
private:
    std::unordered_map<Int, Float> m_; // 一阶矩估计
    std::unordered_map<Int, Float> v_; // 二阶矩估计
    Float beta1_;
    Float beta2_;
    Float epsilon_;
    Int t_; // 当前迭代次数
};

// 为兼容测试的类型别名
using AdaGradOptimizer = AdagradOptimizer;

// 类型转换辅助函数
OptimizerType ParseOptimizerType(const String& type_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_OPTIMIZER_H 