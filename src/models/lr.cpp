#include "models/lr.h"
#include "optimizer.h"
#include "utils/math_util.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <future>  // 添加future支持

namespace simpleflow {

LRModel::LRModel(Int feature_dim, std::shared_ptr<Activation> activation)
    : Model(feature_dim, activation), bias_(0.0), 
      gen_(rd_()), normal_dist_(0.0, 0.001) {
}

void LRModel::Init() {
    InitWeights();
}

void LRModel::InitWeights() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清空现有权重
    weights_.clear();
    
    // 使用小随机值初始化权重
    // 使用正态分布，均值为0，标准差为0.001
    std::normal_distribution<Float> small_init(0.0f, 0.001f);
    
    // 对于特征维度较大的情况，我们采用懒加载策略
    // 这里仅初始化偏置项，特征权重会在需要时动态初始化
    bias_ = small_init(gen_);
    
    // 记录初始化信息
    #ifdef DEBUG
    std::cout << "LR模型初始化: 偏置=" << bias_ << ", 权重使用正态分布N(0,0.001)懒加载初始化" << std::endl;
    #endif
}

Float LRModel::Forward(const SparseFeatureVector& features) {
    // 减少锁的使用，优化性能
    Float sum = bias_; // 偏置可以安全读取，不需要锁
    
    // 只在读取权重时使用细粒度锁
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        
        Float weight = 0.0f;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = weights_.find(index);
            if (it != weights_.end()) {
                weight = it->second;
            }
        }
        
        sum += weight * value;
    }
    
    // 直接应用激活函数，不进行范围限制
    Float activated = activation_->Forward(sum);
    
    return activated;
}

// 添加批量前向传播方法以支持并行处理
void LRModel::Forward(const Batch& batch, FloatVector& predictions) {
    predictions.resize(batch.size());
    
    size_t batch_size = batch.size();
    size_t available_threads = std::thread::hardware_concurrency();
    size_t use_threads = std::min(available_threads, std::max(size_t(1), batch_size / 20));
    
    // 批次太小或系统不支持并发时使用单线程
    if (batch_size < 20 || use_threads <= 1) {
        for (size_t i = 0; i < batch.size(); ++i) {
            predictions[i] = Forward(batch[i].features);
        }
        return;
    }
    
    // 多线程并行处理
    std::vector<std::future<void>> futures;
    size_t samples_per_thread = batch_size / use_threads;
    size_t remainder = batch_size % use_threads;
    
    // 启动多线程处理
    for (size_t t = 0; t < use_threads; ++t) {
        size_t start = t * samples_per_thread + std::min(t, remainder);
        size_t end = start + samples_per_thread + (t < remainder ? 1 : 0);
        
        futures.push_back(std::async(std::launch::async, [this, &batch, &predictions, start, end]() {
            for (size_t i = start; i < end; ++i) {
                predictions[i] = this->Forward(batch[i].features);
            }
        }));
    }
    
    // 等待所有线程完成
    for (auto& future : futures) {
        future.wait();
    }
}

void LRModel::Backward(const SparseFeatureVector& features, 
                      Float label, 
                      Float prediction, 
                      std::shared_ptr<Optimizer> optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("Optimizer cannot be null");
    }
    
    // 计算损失函数的梯度
    Float gradient = CalculateGradient(prediction, label);

    // 计算激活函数的导数
    Float z_derivative = activation_->Backward(prediction);
    // 确保导数不为零
    const Float MIN_DERIVATIVE = 1e-8;
    if (std::abs(z_derivative) < MIN_DERIVATIVE) {
        z_derivative = (z_derivative >= 0) ? MIN_DERIVATIVE : -MIN_DERIVATIVE;
    }
    // 计算最终梯度
    gradient = z_derivative * gradient;

    // 梯度裁剪
    gradient = ClipGradient(gradient);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 更新偏置
    optimizer->Update(0, gradient, bias_);
    
    // 懒加载：确保所有特征的权重都存在
    // 使用小随机值初始化，和InitWeights保持一致
    for (const auto& feature : features) {
        Int index = feature.index;
        if (weights_.find(index) == weights_.end()) {
            // 使用正态分布，均值为0，标准差为0.001，与初始化策略一致
            std::normal_distribution<Float> small_init(0.0f, 0.001f);
            weights_[index] = small_init(gen_);
        }
    }
    
    // 更新特征权重
    for (const auto& feature : features) {
        Int index = feature.index;
        Float value = feature.value;
        Float feat_gradient = gradient * value;
        
        // 梯度裁剪
        feat_gradient = ClipGradient(feat_gradient);
        
        optimizer->Update(index, feat_gradient, weights_[index]);
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