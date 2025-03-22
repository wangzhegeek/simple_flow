#include "activation.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace simpleflow {

void Activation::Forward(const FloatVector& input, FloatVector& output) const {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = Forward(input[i]);
    }
}

void Activation::Gradient(const FloatVector& input, const FloatVector& output, FloatVector& grad) const {
    grad.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        grad[i] = Gradient(input[i], output[i]);
    }
}

Float SigmoidActivation::Forward(Float x) const {
    // 数值稳定版本的sigmoid实现
    // 处理极端值，避免溢出
    if (x > 35.0f) return 1.0f;
    if (x < -35.0f) return 0.0f;
    
    // 通过更稳定的方式计算sigmoid，减少指数溢出风险
    Float result;
    if (x >= 0) {
        Float e = std::exp(-x);
        result = 1.0f / (1.0f + e);
    } else {
        Float e = std::exp(x);
        result = e / (1.0f + e);
    }
    
    return result;
}

Float SigmoidActivation::Gradient(Float x, Float y) const {
    return y * (1.0 - y);
}

Float ReLUActivation::Forward(Float x) const {
    return std::max(0.0f, x);
}

Float ReLUActivation::Gradient(Float x, Float y) const {
    return x > 0.0 ? 1.0 : 0.0;
}

Float TanhActivation::Forward(Float x) const {
    return std::tanh(x);
}

Float TanhActivation::Gradient(Float x, Float y) const {
    return 1.0 - y * y;
}

std::shared_ptr<Activation> Activation::Create(ActivationType type) {
    switch (type) {
        case ActivationType::Identity:
            return std::make_shared<IdentityActivation>();
        case ActivationType::Sigmoid:
            return std::make_shared<SigmoidActivation>();
        case ActivationType::ReLU:
            return std::make_shared<ReLUActivation>();
        case ActivationType::Tanh:
            return std::make_shared<TanhActivation>();
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

ActivationType ParseActivationType(const String& type_str) {
    if (type_str == "identity") {
        return ActivationType::Identity;
    } else if (type_str == "sigmoid") {
        return ActivationType::Sigmoid;
    } else if (type_str == "relu") {
        return ActivationType::ReLU;
    } else if (type_str == "tanh") {
        return ActivationType::Tanh;
    }
    return ActivationType::Unknown;
}

} // namespace simpleflow 