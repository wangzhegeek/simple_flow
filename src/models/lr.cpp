#include "simpleflow/models/lr.h"
#include "simpleflow/utils/math_util.h"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace simpleflow {

LRModel::LRModel(Int feature_dim, std::shared_ptr<Activation> activation)
    : Model(feature_dim, activation), bias_(0.0), 
      gen_(rd_()), normal_dist_(0.0, 0.01) {
}

void LRModel::Init() {
    InitWeights();
}

void LRModel::InitWeights() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    weights_.clear();
    bias_ = 0.0;
}

Float LRModel::Forward(const SparseFeatureVector& features) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Float sum = bias_;
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        auto it = weights_.find(index);
        if (it != weights_.end()) {
            sum += it->second * value;
        }
    }
    
    return activation_->Forward(sum);
}

void LRModel::Backward(const SparseFeatureVector& features, Float gradient, std::shared_ptr<Optimizer> optimizer) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 更新偏置项
    optimizer->Update(0, gradient, bias_);
    
    // 更新特征权重
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        // 如果权重不存在，则初始化
        if (weights_.find(index) == weights_.end()) {
            weights_[index] = normal_dist_(gen_);
        }
        
        // 更新权重
        Float& weight = weights_[index];
        optimizer->Update(index, gradient * value, weight);
    }
}

void LRModel::Save(const String& file_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }
    
    // 保存特征维度
    out.write(reinterpret_cast<const char*>(&feature_dim_), sizeof(feature_dim_));
    
    // 保存偏置项
    out.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
    
    // 保存权重数量
    Int weight_size = weights_.size();
    out.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    
    // 保存权重
    for (const auto& pair : weights_) {
        Int index = pair.first;
        Float value = pair.second;
        
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

void LRModel::Load(const String& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + file_path);
    }
    
    // 加载特征维度
    in.read(reinterpret_cast<char*>(&feature_dim_), sizeof(feature_dim_));
    
    // 加载偏置项
    in.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
    
    // 加载权重数量
    Int weight_size;
    in.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    
    // 加载权重
    weights_.clear();
    for (Int i = 0; i < weight_size; ++i) {
        Int index;
        Float value;
        
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        
        weights_[index] = value;
    }
}

} // namespace simpleflow 