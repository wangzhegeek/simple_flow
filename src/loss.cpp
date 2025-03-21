#include "simpleflow/loss.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace simpleflow {

Float Loss::Compute(const FloatVector& predictions, const FloatVector& targets) const {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets size mismatch");
    }
    
    Float total_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        total_loss += Compute(predictions[i], targets[i]);
    }
    
    return total_loss / predictions.size();
}

void Loss::Gradient(const FloatVector& predictions, const FloatVector& targets, FloatVector& gradients) const {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets size mismatch");
    }
    
    gradients.resize(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        gradients[i] = Gradient(predictions[i], targets[i]);
    }
}

Float MSELoss::Compute(Float prediction, Float target) const {
    Float diff = prediction - target;
    return 0.5 * diff * diff;
}

Float MSELoss::Gradient(Float prediction, Float target) const {
    return prediction - target;
}

Float LogLoss::Compute(Float prediction, Float target) const {
    // 处理可能的-1/+1标签
    Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
    
    // 裁剪预测值，避免数值问题
    Float p = std::max(std::min(prediction, 1.0f - 1e-7f), 1e-7f);
    return -adjusted_target * std::log(p) - (1.0f - adjusted_target) * std::log(1.0f - p);
}

Float LogLoss::Gradient(Float prediction, Float target) const {
    // 处理可能的-1/+1标签
    Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
    
    // 裁剪预测值，避免数值问题
    Float p = std::max(std::min(prediction, 1.0f - 1e-7f), 1e-7f);
    return (p - adjusted_target) / (p * (1.0f - p));
}

Float HingeLoss::Compute(Float prediction, Float target) const {
    // 对于已经是-1/+1的标签，我们不需要再映射
    Float t = (target > 0) ? 1.0f : -1.0f;
    return std::max(0.0f, 1.0f - t * prediction);
}

Float HingeLoss::Gradient(Float prediction, Float target) const {
    // 对于已经是-1/+1的标签，我们不需要再映射
    Float t = (target > 0) ? 1.0f : -1.0f;
    
    if (t * prediction < 1.0f) {
        return -t;
    }
    return 0.0f;
}

std::shared_ptr<Loss> Loss::Create(LossType type) {
    switch (type) {
        case LossType::MSE:
            return std::make_shared<MSELoss>();
        case LossType::LogLoss:
            return std::make_shared<LogLoss>();
        case LossType::Hinge:
            return std::make_shared<HingeLoss>();
        default:
            throw std::runtime_error("Unknown loss type");
    }
}

LossType ParseLossType(const String& type_str) {
    if (type_str == "mse") {
        return LossType::MSE;
    } else if (type_str == "logloss") {
        return LossType::LogLoss;
    } else if (type_str == "hinge") {
        return LossType::Hinge;
    }
    return LossType::Unknown;
}

} // namespace simpleflow 