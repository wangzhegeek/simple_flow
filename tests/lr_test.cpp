#include <gtest/gtest.h>
#include "models/lr.h"
#include "activation.h"
#include "optimizer.h"
#include <memory>
#include <vector>
#include <cmath>

namespace simpleflow {
namespace test {

class LRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建逻辑回归模型
        activation_ = std::make_shared<SigmoidActivation>();
        model_ = std::make_shared<LRModel>(3, activation_);
        model_->Init();
        
        // 设置模型的权重，以便于测试
        std::unordered_map<Int, Float> weights = {{0, 0.1}, {1, 0.2}, {2, 0.3}};
        model_->SetWeights(weights);
        model_->SetBias(0.5);
        
        // 创建测试数据
        features_.push_back(SparseFeature{0, 1.0});
        features_.push_back(SparseFeature{1, 2.0});
        features_.push_back(SparseFeature{2, 3.0});
        
        // 创建优化器
        optimizer_ = std::make_shared<SGDOptimizer>(0.1);
    }
    
    std::shared_ptr<Activation> activation_;
    std::shared_ptr<LRModel> model_;
    SparseFeatureVector features_;
    std::shared_ptr<Optimizer> optimizer_;
};

TEST_F(LRTest, Forward) {
    // 计算前向传播
    Float prediction = model_->Forward(features_);
    
    // 手动计算预期结果
    Float z = 0.5 + 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0;
    Float expected = 1.0 / (1.0 + std::exp(-z));
    
    // 验证预测结果
    EXPECT_FLOAT_EQ(prediction, expected);
}

TEST_F(LRTest, Backward) {
    // 获取当前权重和偏置
    auto original_weights = model_->GetWeights();
    Float original_bias = model_->GetBias();
    
    // 计算前向传播
    Float prediction = model_->Forward(features_);
    
    // 执行反向传播
    Float gradient = 0.5;  // 假设损失函数的梯度为0.5
    model_->Backward(features_, gradient, optimizer_);
    
    // 获取更新后的权重和偏置
    auto updated_weights = model_->GetWeights();
    Float updated_bias = model_->GetBias();
    
    // 验证权重更新
    for (const auto& feature : features_) {
        Int index = feature.index;
        Float value = feature.value;
        Float original = original_weights[index];
        Float expected = original - 0.1 * gradient * value;  // 学习率为0.1
        
        EXPECT_FLOAT_EQ(updated_weights[index], expected);
    }
    
    // 验证偏置更新
    EXPECT_FLOAT_EQ(updated_bias, original_bias - 0.1 * gradient);
}

TEST_F(LRTest, SaveAndLoad) {
    // 保存模型
    std::string model_file = "lr_model_test.dat";
    model_->Save(model_file);
    
    // 创建新模型并加载
    auto new_model = std::make_shared<LRModel>(3, activation_);
    new_model->Load(model_file);
    
    // 验证加载的模型具有相同的权重和偏置
    EXPECT_FLOAT_EQ(new_model->GetBias(), model_->GetBias());
    
    auto original_weights = model_->GetWeights();
    auto loaded_weights = new_model->GetWeights();
    
    for (const auto& pair : original_weights) {
        Int index = pair.first;
        Float weight = pair.second;
        
        EXPECT_FLOAT_EQ(loaded_weights[index], weight);
    }
    
    // 验证两个模型的预测结果相同
    Float pred1 = model_->Forward(features_);
    Float pred2 = new_model->Forward(features_);
    EXPECT_FLOAT_EQ(pred1, pred2);
    
    // 清理测试文件
    std::remove(model_file.c_str());
}

TEST_F(LRTest, Initialization) {
    // 创建新模型并初始化
    auto new_model = std::make_shared<LRModel>(3, activation_);
    new_model->Init();
    
    // 验证初始化后的权重为空（懒加载）
    EXPECT_TRUE(new_model->GetWeights().empty());
    
    // 验证偏置被初始化为0
    EXPECT_FLOAT_EQ(new_model->GetBias(), 0.0);
    
}

} // namespace test
} // namespace simpleflow 