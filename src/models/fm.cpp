#include "simpleflow/models/fm.h"
#include "simpleflow/activation.h"
#include "simpleflow/loss.h"
#include "simpleflow/utils/math_util.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace simpleflow {

FMModel::FMModel(Int feature_dim, Int embedding_size, std::shared_ptr<Activation> activation, std::shared_ptr<Loss> loss)
    : Model(feature_dim, activation),
      embedding_size_(embedding_size),
      bias_(0.0),
      loss_(loss),
      rd_(),
      gen_(rd_()),
      normal_dist_(0.0, 0.01) {
    if (embedding_size <= 0) {
        throw std::invalid_argument("Embedding size must be positive");
    }
    Init();
}

void FMModel::Init() {
    // 初始化权重
    InitWeights();
}

void FMModel::InitWeights() {
    // 初始化偏置项
    bias_ = 0.0;
    
    // 初始化一阶权重 - 使用懒惰初始化，当需要时才创建
    weights_.clear();
    
    // 初始化二阶特征嵌入 - 同样懒惰初始化
    embeddings_.clear();
}

Float FMModel::Forward(const SparseFeatureVector& features) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Float result = bias_;
    
    // 线性部分
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        auto it = weights_.find(index);
        if (it != weights_.end()) {
            result += it->second * value;
        }
    }

    // 二阶交互部分
    Float interaction = ComputeSecondOrderInteraction(features);
    result += interaction;
    
    // 应用激活函数
    Float activated = activation_->Forward(result);
    
    // 防止数值问题
    if (std::isnan(activated) || std::isinf(activated)) {
        return 0.5; // 默认返回中间值
    }
    
    return activated;
}

Float FMModel::ComputeSecondOrderInteraction(const SparseFeatureVector& features) {
    // 使用公式: 0.5 * sum((sum_i v_i * x_i)^2 - sum_i (v_i * x_i)^2)
    
    // 为了节省计算，我们预先计算向量的和和平方和
    std::vector<Float> sum_vector(embedding_size_, 0.0);
    std::vector<Float> sum_square_vector(embedding_size_, 0.0);

    // 计数有效特征数量
    int valid_features = 0;
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        // 如果没有嵌入，可能是稀疏特征，初始化一个嵌入
        auto it = embeddings_.find(index);
        if (it == embeddings_.end()) {
            // 懒惰初始化 - 通常不会执行到这里，除非前向传播在反向传播之前
            // 使用Xavier初始化
            Float xavier_range = std::sqrt(6.0 / (feature_dim_ + embedding_size_));
            embeddings_[index].resize(embedding_size_);
            for (Int f = 0; f < embedding_size_; ++f) {
                embeddings_[index][f] = (2.0 * static_cast<Float>(std::rand()) / RAND_MAX - 1.0) * xavier_range;
            }
            it = embeddings_.find(index);
        }
        
        const std::vector<Float>& embedding = it->second;
        valid_features++;
        
        for (Int f = 0; f < embedding_size_; ++f) {
            Float feat_embed = value * embedding[f];
            sum_vector[f] += feat_embed;
            sum_square_vector[f] += feat_embed * feat_embed;
        }
    }
    
    // 如果没有有效特征，返回0
    if (valid_features == 0) {
        return 0.0;
    }
    
    // 计算最终的交互项
    Float sum_square = 0.0;
    Float square_sum = 0.0;
    
    for (Int f = 0; f < embedding_size_; ++f) {
        sum_square += sum_vector[f] * sum_vector[f];
        square_sum += sum_square_vector[f];
    }
    
    // 增加数值稳定性，确保结果不会太大
    Float result = 0.5 * (sum_square - square_sum);
    
    // 防止数值问题
    if (std::isnan(result) || std::isinf(result)) {
        return 0.0;
    }
    
    // 缩放交互项，使其与线性项在同一量级
    result *= 0.01;
    
    return result;
}

void FMModel::Backward(const SparseFeatureVector& features, Float gradient, std::shared_ptr<Optimizer> optimizer) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 梯度裁剪，避免梯度爆炸
    const Float grad_clip = 1.0;
    if (gradient > grad_clip) gradient = grad_clip;
    if (gradient < -grad_clip) gradient = -grad_clip;
    
    // 检查梯度是否为NaN或Inf
    if (std::isnan(gradient) || std::isinf(gradient)) {
        return;  // 如果梯度有问题，直接不更新
    }
    
    // 更新偏置项
    optimizer->Update(0, gradient, bias_);
    
    // 更新线性权重
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        // 如果权重不存在，则初始化
        if (weights_.find(index) == weights_.end()) {
            weights_[index] = normal_dist_(gen_) * 0.01;
        }
        
        // 更新权重
        Float& weight = weights_[index];
        Float feat_gradient = gradient * value;
        
        // 梯度裁剪
        if (feat_gradient > grad_clip) feat_gradient = grad_clip;
        if (feat_gradient < -grad_clip) feat_gradient = -grad_clip;
        
        optimizer->Update(index, feat_gradient, weight);
    }
    
    // 为了计算二阶部分的梯度，计算各种中间结果
    std::vector<Float> sum_vector(embedding_size_, 0.0);
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        auto it = embeddings_.find(index);
        if (it == embeddings_.end()) {
            Float xavier_range = std::sqrt(6.0 / (feature_dim_ + embedding_size_));
            embeddings_[index].resize(embedding_size_);
            for (Int f = 0; f < embedding_size_; ++f) {
                embeddings_[index][f] = (2.0 * static_cast<Float>(std::rand()) / RAND_MAX - 1.0) * xavier_range;
            }
            it = embeddings_.find(index);
        }
        
        const std::vector<Float>& embedding = it->second;
        for (Int f = 0; f < embedding_size_; ++f) {
            sum_vector[f] += value * embedding[f];
        }
    }
    
    // 更新二阶嵌入向量
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        std::vector<Float>& embedding = embeddings_[index];
        
        for (Int f = 0; f < embedding_size_; ++f) {
            // 计算二阶梯度
            Float sum_except_i = sum_vector[f] - value * embedding[f];
            Float grad_embedding = value * sum_except_i * 0.01;
            
            // 应用梯度
            Float embed_gradient = gradient * grad_embedding;
            
            // 梯度裁剪
            if (embed_gradient > grad_clip) embed_gradient = grad_clip;
            if (embed_gradient < -grad_clip) embed_gradient = -grad_clip;
            
            // 检查梯度是否为NaN或Inf
            if (std::isnan(embed_gradient) || std::isinf(embed_gradient)) {
                continue;
            }
            
            // 唯一索引：线性主键 + 嵌入维度
            Int unique_index = index * embedding_size_ + f + feature_dim_;
            optimizer->Update(unique_index, embed_gradient, embedding[f]);
        }
    }
}

void FMModel::Save(const String& file_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }
    
    // 保存特征维度和嵌入维度
    out.write(reinterpret_cast<const char*>(&feature_dim_), sizeof(feature_dim_));
    out.write(reinterpret_cast<const char*>(&embedding_size_), sizeof(embedding_size_));
    
    // 保存偏置项
    out.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
    
    // 保存一阶权重数量
    Int weight_size = weights_.size();
    out.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    
    // 保存一阶权重
    for (const auto& pair : weights_) {
        Int index = pair.first;
        Float value = pair.second;
        
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    
    // 保存二阶嵌入向量数量
    Int embedding_size = embeddings_.size();
    out.write(reinterpret_cast<const char*>(&embedding_size), sizeof(embedding_size));
    
    // 保存二阶嵌入向量
    for (const auto& pair : embeddings_) {
        Int index = pair.first;
        const std::vector<Float>& embedding = pair.second;
        
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        out.write(reinterpret_cast<const char*>(embedding.data()), sizeof(Float) * embedding_size_);
    }
}

void FMModel::Load(const String& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + file_path);
    }
    
    // 加载特征维度和嵌入维度
    in.read(reinterpret_cast<char*>(&feature_dim_), sizeof(feature_dim_));
    in.read(reinterpret_cast<char*>(&embedding_size_), sizeof(embedding_size_));
    
    // 加载偏置项
    in.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
    
    // 加载一阶权重数量
    Int weight_size;
    in.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    
    // 加载一阶权重
    weights_.clear();
    for (Int i = 0; i < weight_size; ++i) {
        Int index;
        Float value;
        
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        
        weights_[index] = value;
    }
    
    // 加载二阶嵌入向量数量
    Int embedding_size;
    in.read(reinterpret_cast<char*>(&embedding_size), sizeof(embedding_size));
    
    // 加载二阶嵌入向量
    embeddings_.clear();
    for (Int i = 0; i < embedding_size; ++i) {
        Int index;
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        
        embeddings_[index].resize(embedding_size_);
        in.read(reinterpret_cast<char*>(embeddings_[index].data()), sizeof(Float) * embedding_size_);
    }
}

} // namespace simpleflow 