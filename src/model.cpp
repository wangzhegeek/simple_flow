#include "model.h"
#include "models/lr.h"
#include "models/fm.h"
#include <fstream>
#include <stdexcept>
#include <typeinfo>

namespace simpleflow {

Model::Model(Int feature_dim, std::shared_ptr<Activation> activation)
    : feature_dim_(feature_dim), activation_(activation) {
    if (feature_dim <= 0) {
        throw std::invalid_argument("Feature dimension must be positive");
    }
    
    if (!activation) {
        throw std::invalid_argument("Activation function cannot be null");
    }
}

void Model::Forward(const Batch& batch, FloatVector& predictions) {
    predictions.resize(batch.size());
    
    for (size_t i = 0; i < batch.size(); ++i) {
        predictions[i] = Forward(batch[i].features);
    }
}

void Model::Backward(const Batch& batch, const FloatVector& gradients, std::shared_ptr<Optimizer> optimizer) {
    if (batch.size() != gradients.size()) {
        throw std::runtime_error("Batch size and gradients size mismatch");
    }
    
    // 先获取预测值
    FloatVector predictions;
    predictions.resize(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        predictions[i] = Forward(batch[i].features);
    }
    
    // 对每个样本执行反向传播
    for (size_t i = 0; i < batch.size(); ++i) {
        // 统一使用新的反向传播接口
        Backward(batch[i].features, batch[i].label, predictions[i], optimizer);
    }
}

void Model::Save(const String& file_path) const {
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }
    
    // 保存特征维度
    out.write(reinterpret_cast<const char*>(&feature_dim_), sizeof(feature_dim_));
    
    // 子类需要实现自己的保存逻辑
}

void Model::Load(const String& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + file_path);
    }
    
    // 加载特征维度
    in.read(reinterpret_cast<char*>(&feature_dim_), sizeof(feature_dim_));
    
    // 子类需要实现自己的加载逻辑
}

ModelType ParseModelType(const String& type_str) {
    String lower_type = type_str;
    for (auto& c : lower_type) {
        c = std::tolower(c);
    }
    
    if (lower_type == "lr" || lower_type == "logistic_regression" || lower_type == "logisticregression") {
        return ModelType::LR;
    } else if (lower_type == "fm" || lower_type == "factorization_machine" || lower_type == "factorizationmachine") {
        return ModelType::FM;
    }
    return ModelType::Unknown;
}

std::shared_ptr<Model> Model::Create(ModelType type, Int feature_dim, const std::unordered_map<String, String>& params) {
    switch (type) {
        case ModelType::LR: {
            auto activation = std::make_shared<SigmoidActivation>();
            return std::make_shared<LRModel>(feature_dim, activation);
        }
        case ModelType::FM: {
            auto activation = std::make_shared<SigmoidActivation>();
            Int embedding_size = 8;  // 默认嵌入维度
            
            auto it = params.find("embedding_size");
            if (it != params.end()) {
                embedding_size = std::stoi(it->second);
            }
            
            return std::make_shared<FMModel>(feature_dim, embedding_size, activation);
        }
        default:
            throw std::runtime_error("Unknown model type");
    }
}

Float Model::CalculateGradient(Float prediction, Float label) const {
    // 防止数值溢出
    if (prediction > 1.0f - constants::EPSILON) {
        prediction = 1.0f - constants::EPSILON;
    }
    if (prediction < constants::EPSILON) {
        prediction = constants::EPSILON;
    }
    
    // 计算二分类对数损失梯度 (p - y)
    return prediction - label;
}

} // namespace simpleflow 