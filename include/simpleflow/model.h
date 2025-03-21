#ifndef SIMPLEFLOW_MODEL_H
#define SIMPLEFLOW_MODEL_H

#include "simpleflow/types.h"
#include "simpleflow/optimizer.h"
#include "simpleflow/activation.h"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace simpleflow {

// 模型基类
class Model {
public:
    Model(Int feature_dim, std::shared_ptr<Activation> activation);
    virtual ~Model() = default;
    
    // 初始化模型参数
    virtual void Init() = 0;
    
    // 前向传播
    virtual Float Forward(const SparseFeatureVector& features) = 0;
    
    // 批量前向传播
    virtual void Forward(const Batch& batch, FloatVector& predictions);
    
    // 反向传播更新参数
    virtual void Backward(const SparseFeatureVector& features, Float gradient, std::shared_ptr<Optimizer> optimizer) = 0;
    
    // 批量反向传播
    virtual void Backward(const Batch& batch, const FloatVector& gradients, std::shared_ptr<Optimizer> optimizer);
    
    // 保存模型
    virtual void Save(const String& file_path) const;
    
    // 加载模型
    virtual void Load(const String& file_path);
    
    // 获取特征维度
    Int GetFeatureDim() const { return feature_dim_; }
    
    // 获取激活函数
    std::shared_ptr<Activation> GetActivation() const { return activation_; }
    
    // 设置激活函数
    void SetActivation(std::shared_ptr<Activation> activation) { activation_ = activation; }
    
    // 工厂方法创建模型（带参数）
    static std::shared_ptr<Model> Create(ModelType type, Int feature_dim, const std::unordered_map<String, String>& params);
    
    // 工厂方法创建模型（简化版，用于兼容测试）
    static std::shared_ptr<Model> Create(ModelType type, Int feature_dim) {
        if (type == ModelType::Unknown) {
            throw std::runtime_error("Unknown model type");
        }
        return Create(type, feature_dim, {});
    }
    
protected:
    Int feature_dim_;
    std::shared_ptr<Activation> activation_;
};

// 类型转换辅助函数（用于兼容测试）
ModelType ParseModelType(const String& type_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_MODEL_H 