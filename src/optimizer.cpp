#include "simpleflow/optimizer.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace simpleflow {

// 梯度裁剪阈值 - 改为更大的值，避免过度限制梯度
const Float GRADIENT_CLIP_THRESHOLD = 5.0f;

Optimizer::Optimizer(Float learning_rate, Float l2_reg)
    : learning_rate_(learning_rate), l2_reg_(l2_reg) {
    if (learning_rate <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

// 向量级更新的基类实现
void Optimizer::Update(FloatVector& parameters, const FloatVector& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::runtime_error("Parameters and gradients size mismatch");
    }
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        Update(i, gradients[i], parameters[i]);
    }
}

// 梯度裁剪辅助函数
Float ClipGradient(Float gradient) {
    // 处理NaN或Inf
    if (std::isnan(gradient) || std::isinf(gradient)) {
        return 0.0f;
    }
    
    // 裁剪梯度值
    return std::max(std::min(gradient, GRADIENT_CLIP_THRESHOLD), -GRADIENT_CLIP_THRESHOLD);
}

// SGD优化器
SGDOptimizer::SGDOptimizer(Float learning_rate, Float l2_reg)
    : Optimizer(learning_rate, l2_reg) {
}

void SGDOptimizer::Update(Int index, Float gradient, Float& parameter) {
    // L2正则化梯度
    gradient += l2_reg_ * parameter;
    
    // 梯度裁剪
    gradient = ClipGradient(gradient);
    
    // 参数更新
    parameter -= learning_rate_ * gradient;
}

// Adagrad优化器
AdagradOptimizer::AdagradOptimizer(Float learning_rate, Float l2_reg, Float epsilon)
    : Optimizer(learning_rate, l2_reg), epsilon_(epsilon) {
}

void AdagradOptimizer::Update(Int index, Float gradient, Float& parameter) {
    // L2正则化梯度
    gradient += l2_reg_ * parameter;
    
    // 梯度裁剪
    gradient = ClipGradient(gradient);
    
    // 累积平方梯度
    Float& squared_gradient = squared_gradients_[index];
    squared_gradient += gradient * gradient;
    
    // 参数更新
    parameter -= learning_rate_ * gradient / (std::sqrt(squared_gradient) + epsilon_);
}

// RMSProp优化器
RMSPropOptimizer::RMSPropOptimizer(Float learning_rate, Float l2_reg, Float decay_rate, Float epsilon)
    : Optimizer(learning_rate, l2_reg), decay_rate_(decay_rate), epsilon_(epsilon) {
}

void RMSPropOptimizer::Update(Int index, Float gradient, Float& parameter) {
    // L2正则化梯度
    gradient += l2_reg_ * parameter;
    
    // 梯度裁剪
    gradient = ClipGradient(gradient);
    
    // 更新缓存
    Float& cache = cache_[index];
    cache = decay_rate_ * cache + (1 - decay_rate_) * gradient * gradient;
    
    // 参数更新
    parameter -= learning_rate_ * gradient / (std::sqrt(cache) + epsilon_);
}

// Adam优化器
AdamOptimizer::AdamOptimizer(Float learning_rate, Float l2_reg, Float beta1, Float beta2, Float epsilon)
    : Optimizer(learning_rate, l2_reg), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
}

void AdamOptimizer::Update(Int index, Float gradient, Float& parameter) {
    // 应用L2正则化
    gradient += l2_reg_ * parameter;
    
    // 梯度裁剪
    gradient = ClipGradient(gradient);
    
    // 单个参数更新时也需要更新迭代计数器
    ++t_;
    
    // 更新一阶矩估计
    Float& m = m_[index];
    m = beta1_ * m + (1 - beta1_) * gradient;
    
    // 更新二阶矩估计
    Float& v = v_[index];
    v = beta2_ * v + (1 - beta2_) * gradient * gradient;
    
    // 修正一阶矩的偏差
    Float m_hat = m / (1 - std::pow(beta1_, t_));
    
    // 修正二阶矩的偏差
    Float v_hat = v / (1 - std::pow(beta2_, t_));
    
    // 参数更新
    parameter -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
}

// 重写向量Update方法，确保t_计数器只增加一次
void AdamOptimizer::Update(FloatVector& parameters, const FloatVector& gradients) {
    if (parameters.size() != gradients.size()) {
        throw std::runtime_error("Parameters and gradients size mismatch");
    }
    
    // 先增加迭代计数器，确保所有参数使用相同的t_值
    ++t_;
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        Float gradient = gradients[i] + l2_reg_ * parameters[i];
        
        // 梯度裁剪
        gradient = ClipGradient(gradient);
        
        // 更新一阶矩估计
        Float& m = m_[i];
        m = beta1_ * m + (1 - beta1_) * gradient;
        
        // 更新二阶矩估计
        Float& v = v_[i];
        v = beta2_ * v + (1 - beta2_) * gradient * gradient;
        
        // 修正一阶矩的偏差
        Float m_hat = m / (1 - std::pow(beta1_, t_));
        
        // 修正二阶矩的偏差
        Float v_hat = v / (1 - std::pow(beta2_, t_));
        
        // 参数更新
        parameters[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

// 工厂方法创建优化器
std::shared_ptr<Optimizer> Optimizer::Create(OptimizerType type, Float learning_rate, Float l2_reg) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_shared<SGDOptimizer>(learning_rate, l2_reg);
        case OptimizerType::Adagrad:
            return std::make_shared<AdagradOptimizer>(learning_rate, l2_reg);
        case OptimizerType::Adam:
            return std::make_shared<AdamOptimizer>(learning_rate, l2_reg);
        case OptimizerType::RMSProp:
            return std::make_shared<RMSPropOptimizer>(learning_rate, l2_reg);
        default:
            throw std::runtime_error("Unknown optimizer type");
    }
}

// 类型转换辅助函数
OptimizerType ParseOptimizerType(const String& type_str) {
    String lower_type = type_str;
    for (auto& c : lower_type) {
        c = std::tolower(c);
    }
    
    if (lower_type == "sgd") {
        return OptimizerType::SGD;
    } else if (lower_type == "adagrad") {
        return OptimizerType::Adagrad;
    } else if (lower_type == "adam") {
        return OptimizerType::Adam;
    } else if (lower_type == "rmsprop") {
        return OptimizerType::RMSProp;
    }
    return OptimizerType::Unknown;
}

} // namespace simpleflow 