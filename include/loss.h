#ifndef SIMPLEFLOW_LOSS_H
#define SIMPLEFLOW_LOSS_H

#include "types.h"
#include <memory>

namespace simpleflow {

// 损失函数基类
class Loss {
public:
    virtual ~Loss() = default;
    
    // 计算损失
    virtual Float Compute(Float prediction, Float target) const = 0;
    
    // 计算一批数据的平均损失
    virtual Float Compute(const FloatVector& predictions, const FloatVector& targets) const;
    
    // 计算梯度
    virtual Float Gradient(Float prediction, Float target) const = 0;
    
    // 计算一批数据的梯度
    virtual void Gradient(const FloatVector& predictions, const FloatVector& targets, FloatVector& gradients) const;
    
    // Backward方法 - 作为Gradient的别名，方便使用
    virtual Float Backward(Float prediction, Float target) const {
        return Gradient(prediction, target);
    }
    
    // 工厂方法创建损失函数
    static std::shared_ptr<Loss> Create(LossType type);
};

// 均方误差损失
class MSELoss : public Loss {
public:
    Float Compute(Float prediction, Float target) const override;
    Float Gradient(Float prediction, Float target) const override;
};

// 对数损失（用于二分类）
class LogLoss : public Loss {
public:
    Float Compute(Float prediction, Float target) const override;
    Float Gradient(Float prediction, Float target) const override;
};

// Hinge损失（用于SVM）
class HingeLoss : public Loss {
public:
    Float Compute(Float prediction, Float target) const override;
    Float Gradient(Float prediction, Float target) const override;
};

// 类型转换辅助函数
LossType ParseLossType(const String& type_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_LOSS_H 