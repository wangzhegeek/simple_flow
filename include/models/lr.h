#ifndef SIMPLEFLOW_MODELS_LR_H
#define SIMPLEFLOW_MODELS_LR_H

#include "../model.h"
#include <unordered_map>
#include <random>
#include <mutex>

namespace simpleflow {

// 逻辑回归模型类
class LRModel : public Model {
public:
    LRModel(Int feature_dim, std::shared_ptr<Activation> activation);
    ~LRModel() override = default;
    
    // 初始化模型参数
    void Init() override;
    
    // 前向传播
    Float Forward(const SparseFeatureVector& features) override;
    
    // 批量前向传播（优化版本）
    void Forward(const Batch& batch, FloatVector& predictions) override;
    
    // 统一的反向传播接口
    void Backward(const SparseFeatureVector& features, 
                 Float label, 
                 Float prediction, 
                 std::shared_ptr<Optimizer> optimizer) override;
    
    // 保存模型
    void Save(const String& file_path) const override;
    
    // 加载模型
    void Load(const String& file_path) override;
    
    // 获取和设置权重（用于测试）
    const std::unordered_map<Int, Float>& GetWeights() const { return weights_; }
    void SetWeights(const std::unordered_map<Int, Float>& weights) { weights_ = weights; }
    
    // 获取和设置偏置（用于测试）
    Float GetBias() const { return bias_; }
    void SetBias(Float bias) { bias_ = bias; }
    
private:
    // 模型参数
    std::unordered_map<Int, Float> weights_;
    Float bias_;
    
    // 参数初始化
    void InitWeights();
    
    // 随机数生成器
    std::random_device rd_;
    std::mt19937 gen_;
    std::normal_distribution<Float> normal_dist_;
    
    // 线程安全
    mutable std::mutex mutex_;
};

} // namespace simpleflow

#endif // SIMPLEFLOW_MODELS_LR_H 