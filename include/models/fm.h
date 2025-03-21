#ifndef SIMPLEFLOW_MODELS_FM_H
#define SIMPLEFLOW_MODELS_FM_H

#include "../model.h"
#include "../loss.h"
#include <unordered_map>
#include <random>
#include <mutex>

namespace simpleflow {

// 因子分解机模型类
class FMModel : public Model {
public:
    FMModel(Int feature_dim, Int embedding_size, std::shared_ptr<Activation> activation, std::shared_ptr<Loss> loss = nullptr);
    ~FMModel() override = default;
    
    // 初始化模型参数
    void Init() override;
    
    // 前向传播
    Float Forward(const SparseFeatureVector& features) override;
    
    // 反向传播更新参数
    void Backward(const SparseFeatureVector& features, Float pred, Float target, Optimizer* optimizer);
    
    // 原来的接口，保持兼容性
    void Backward(const SparseFeatureVector& features, Float gradient, std::shared_ptr<Optimizer> optimizer) override;
    
    // 保存模型
    void Save(const String& file_path) const override;
    
    // 加载模型
    void Load(const String& file_path) override;
    
    // 获取嵌入维度
    Int GetEmbeddingSize() const { return embedding_size_; }
    
    // 获取和设置权重（用于测试）
    const std::unordered_map<Int, Float>& GetWeights() const { return weights_; }
    void SetWeights(const std::unordered_map<Int, Float>& weights) { weights_ = weights; }
    
    // 获取和设置因子（用于测试）
    const std::unordered_map<Int, std::vector<Float>>& GetFactors() const { return embeddings_; }
    void SetFactors(const std::unordered_map<Int, std::vector<Float>>& factors) { embeddings_ = factors; }
    
    // 获取和设置偏置（用于测试）
    Float GetBias() const { return bias_; }
    void SetBias(Float bias) { bias_ = bias; }
    
private:
    // 模型参数
    std::unordered_map<Int, Float> weights_; // 一阶权重
    std::unordered_map<Int, std::vector<Float>> embeddings_; // 二阶特征嵌入
    Float bias_; // 偏置项
    Int embedding_size_; // 嵌入维度
    std::shared_ptr<Loss> loss_; // 损失函数
    
    // 参数初始化
    void InitWeights();
    
    // 随机数生成器
    std::random_device rd_;
    std::mt19937 gen_;
    std::normal_distribution<Float> normal_dist_;
    
    // 线程安全
    mutable std::mutex mutex_;
    
    // 计算二阶特征交互
    Float ComputeSecondOrderInteraction(const SparseFeatureVector& features);
};

} // namespace simpleflow

#endif // SIMPLEFLOW_MODELS_FM_H 