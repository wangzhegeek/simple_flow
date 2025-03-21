#ifndef SIMPLEFLOW_ACTIVATION_H
#define SIMPLEFLOW_ACTIVATION_H

#include "types.h"
#include <memory>

namespace simpleflow {

// 激活函数基类
class Activation {
public:
    virtual ~Activation() = default;
    
    // 前向传播
    virtual Float Forward(Float x) const = 0;
    
    // 对一组值进行前向传播
    virtual void Forward(const FloatVector& input, FloatVector& output) const;
    
    // 计算梯度
    virtual Float Gradient(Float x, Float y) const = 0;
    
    // 对一组值计算梯度
    virtual void Gradient(const FloatVector& input, const FloatVector& output, FloatVector& grad) const;
    
    // 反向传播计算梯度 (此方法用于直接获取激活函数的导数)
    virtual Float Backward(Float y) const {
        // 默认实现，派生类可以覆盖以提高效率
        return Gradient(0.0, y);
    }
    
    // 工厂方法创建激活函数
    static std::shared_ptr<Activation> Create(ActivationType type);
};

// Identity激活函数
class IdentityActivation : public Activation {
public:
    Float Forward(Float x) const override { return x; }
    Float Gradient(Float x, Float y) const override { return 1.0; }
    Float Backward(Float y) const override { return 1.0; }
};

// Sigmoid激活函数
class SigmoidActivation : public Activation {
public:
    Float Forward(Float x) const override;
    Float Gradient(Float x, Float y) const override;
    Float Backward(Float y) const override { return y * (1.0 - y); }
};

// ReLU激活函数
class ReLUActivation : public Activation {
public:
    Float Forward(Float x) const override;
    Float Gradient(Float x, Float y) const override;
    Float Backward(Float y) const override { return y > 0.0 ? 1.0 : 0.0; }
};

// Tanh激活函数
class TanhActivation : public Activation {
public:
    Float Forward(Float x) const override;
    Float Gradient(Float x, Float y) const override;
    Float Backward(Float y) const override { return 1.0 - y * y; }
};

// 类型转换辅助函数
ActivationType ParseActivationType(const String& type_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_ACTIVATION_H 