#ifndef SIMPLEFLOW_TYPES_H
#define SIMPLEFLOW_TYPES_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace simpleflow {

// 基本数据类型
using Float = float;
using Int = int;
using String = std::string;
using FloatVector = std::vector<Float>;
using IntVector = std::vector<Int>;
using StringVector = std::vector<String>;

// 稀疏特征
struct SparseFeature {
    Int index;
    Float value;
    
    SparseFeature() : index(0), value(0) {}
    SparseFeature(Int idx, Float val) : index(idx), value(val) {}
};

using SparseFeatureVector = std::vector<SparseFeature>;

// 样本
struct Sample {
    Float label;
    SparseFeatureVector features;
    
    Sample() : label(0) {}
};

using SampleVector = std::vector<Sample>;
using Batch = SampleVector;

// 模型类型
enum class ModelType {
    LR,
    FM,
    LogisticRegression = LR,  // 兼容测试的别名
    Unknown
};

// 激活函数类型
enum class ActivationType {
    Identity,
    Sigmoid,
    ReLU,
    Tanh,
    Unknown
};

// 损失函数类型
enum class LossType {
    MSE,
    LogLoss,
    Hinge,
    Unknown
};

// 优化器类型
enum class OptimizerType {
    SGD,
    Adagrad,
    AdaGrad = Adagrad,  // 兼容测试
    Adam,
    RMSProp,            // 添加RMSProp类型
    Unknown
};

} // namespace simpleflow

#endif // SIMPLEFLOW_TYPES_H 