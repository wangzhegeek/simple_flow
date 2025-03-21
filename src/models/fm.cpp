#include "models/fm.h"
#include "activation.h"
#include "loss.h"
#include "utils/math_util.h"
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
    InitWeights();
}

void FMModel::InitWeights() {
    bias_ = 0.0;
    weights_.clear();
    embeddings_.clear();
}

Float FMModel::Forward(const SparseFeatureVector& features) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Float linear_term = bias_;
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        auto it = weights_.find(index);
        if (it != weights_.end()) {
            linear_term += it->second * value;
        }
    }

    Float interaction = ComputeSecondOrderInteraction(features);
    
    const Float MAX_SCORE = 15.0;
    Float raw_score = linear_term + interaction;
    if (raw_score > MAX_SCORE) raw_score = MAX_SCORE;
    if (raw_score < -MAX_SCORE) raw_score = -MAX_SCORE;
    
    Float activated = activation_->Forward(raw_score);
    
    const Float EPSILON = 1e-10;
    if (activated > 1.0 - EPSILON) activated = 1.0 - EPSILON;
    if (activated < EPSILON) activated = EPSILON;
    
    return activated;
}

Float FMModel::ComputeSecondOrderInteraction(const SparseFeatureVector& features) {
    std::vector<Float> sum_vector(embedding_size_, 0.0);
    std::vector<Float> sum_square_vector(embedding_size_, 0.0);

    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        auto it = embeddings_.find(index);
        if (it == embeddings_.end()) {
            embeddings_[index].resize(embedding_size_);
            for (Int f = 0; f < embedding_size_; ++f) {
                embeddings_[index][f] = normal_dist_(gen_);
            }
            it = embeddings_.find(index);
        }
        
        const std::vector<Float>& embedding = it->second;
        
        for (Int f = 0; f < embedding_size_; ++f) {
            Float feat_embed = value * embedding[f];
            sum_vector[f] += feat_embed;
            sum_square_vector[f] += feat_embed * feat_embed;
        }
    }
    
    Float interaction = 0.0;
    
    for (Int f = 0; f < embedding_size_; ++f) {
        interaction += 0.5 * (sum_vector[f] * sum_vector[f] - sum_square_vector[f]);
    }
    
    return interaction;
}

void FMModel::Backward(const SparseFeatureVector& features, Float pred, Float target, Optimizer* optimizer) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (target == -1) {
        target = 0;
    }
    
    const Float EPSILON = 1e-10;
    if (pred > 1.0 - EPSILON) pred = 1.0 - EPSILON;
    if (pred < EPSILON) pred = EPSILON;
    
    Float diff = loss_->Gradient(pred, target);
    
    if (std::isnan(diff) || std::isinf(diff)) {
        return;
    }
    
    Float z_derivative = activation_->Backward(pred);
    
    const Float MIN_DERIVATIVE = 1e-8;
    if (std::abs(z_derivative) < MIN_DERIVATIVE) {
        z_derivative = (z_derivative >= 0) ? MIN_DERIVATIVE : -MIN_DERIVATIVE;
    }
    
    Float z = z_derivative * diff;
    
    const Float grad_clip = 1.0;
    if (z > grad_clip) z = grad_clip;
    if (z < -grad_clip) z = -grad_clip;
    
    optimizer->Update(0, z * 0.1, bias_);
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        if (weights_.find(index) == weights_.end()) {
            weights_[index] = normal_dist_(gen_) * 0.01;
        }
        
        Float& weight = weights_[index];
        Float feat_gradient = z * value;
        
        if (feat_gradient > grad_clip) feat_gradient = grad_clip;
        if (feat_gradient < -grad_clip) feat_gradient = -grad_clip;
        
        optimizer->Update(index, feat_gradient, weight);
    }
    
    std::vector<Float> sum_vector(embedding_size_, 0.0);
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        auto it = embeddings_.find(index);
        if (it == embeddings_.end()) {
            embeddings_[index].resize(embedding_size_);
            for (Int f = 0; f < embedding_size_; ++f) {
                embeddings_[index][f] = normal_dist_(gen_) * 0.001;
            }
            it = embeddings_.find(index);
        }
        
        const std::vector<Float>& embedding = it->second;
        for (Int f = 0; f < embedding_size_; ++f) {
            sum_vector[f] += value * embedding[f];
        }
    }
    
    const Float embed_scale = 0.1;
    
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        std::vector<Float>& embedding = embeddings_[index];
        
        for (Int f = 0; f < embedding_size_; ++f) {
            Float sum_except_i = sum_vector[f] - value * embedding[f];
            Float grad_embedding = value * sum_except_i;
            
            Float embed_gradient = z * grad_embedding * embed_scale;
            
            if (embed_gradient > grad_clip) embed_gradient = grad_clip;
            if (embed_gradient < -grad_clip) embed_gradient = -grad_clip;
            
            Int unique_index = feature_dim_ + index * embedding_size_ + f;
            optimizer->Update(unique_index, embed_gradient, embedding[f]);
            
            if (embedding[f] > 3.0) embedding[f] = 3.0;
            if (embedding[f] < -3.0) embedding[f] = -3.0;
        }
    }
}

void FMModel::Backward(const SparseFeatureVector& features, Float gradient, std::shared_ptr<Optimizer> optimizer) {
    Float fake_pred = gradient;
    Float fake_target = 0.0;
    
    Backward(features, fake_pred, fake_target, optimizer.get());
}

void FMModel::Save(const String& file_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }
    
    out.write(reinterpret_cast<const char*>(&feature_dim_), sizeof(feature_dim_));
    out.write(reinterpret_cast<const char*>(&embedding_size_), sizeof(embedding_size_));
    out.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
    
    Int weight_size = weights_.size();
    out.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
    
    for (const auto& pair : weights_) {
        Int index = pair.first;
        Float value = pair.second;
        
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    
    Int embedding_size = embeddings_.size();
    out.write(reinterpret_cast<const char*>(&embedding_size), sizeof(embedding_size));
    
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
    
    in.read(reinterpret_cast<char*>(&feature_dim_), sizeof(feature_dim_));
    in.read(reinterpret_cast<char*>(&embedding_size_), sizeof(embedding_size_));
    in.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
    
    Int weight_size;
    in.read(reinterpret_cast<char*>(&weight_size), sizeof(weight_size));
    
    weights_.clear();
    for (Int i = 0; i < weight_size; ++i) {
        Int index;
        Float value;
        
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        
        weights_[index] = value;
    }
    
    Int embedding_size;
    in.read(reinterpret_cast<char*>(&embedding_size), sizeof(embedding_size));
    
    embeddings_.clear();
    for (Int i = 0; i < embedding_size; ++i) {
        Int index;
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        
        embeddings_[index].resize(embedding_size_);
        in.read(reinterpret_cast<char*>(embeddings_[index].data()), sizeof(Float) * embedding_size_);
    }
}

} // namespace simpleflow 