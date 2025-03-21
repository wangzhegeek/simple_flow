#include <gtest/gtest.h>
#include "simpleflow/models/fm.h"
#include "simpleflow/activation.h"
#include "simpleflow/optimizer.h"
#include <memory>
#include <vector>
#include <cmath>

namespace simpleflow {
namespace test {

class FMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建FM模型，设置特征维度为3，嵌入维度为2
        activation_ = std::make_shared<SigmoidActivation>();
        model_ = std::make_shared<FMModel>(3, 2, activation_);
        model_->Init();
        
        // 设置模型的权重，以便于测试
        // 线性部分
        std::unordered_map<Int, Float> weights = {{0, 0.1}, {1, 0.2}, {2, 0.3}};
        model_->SetWeights(weights);
        model_->SetBias(0.5);
        
        // 因子部分
        std::unordered_map<Int, std::vector<Float>> factors;
        factors[0] = {0.1, 0.2};
        factors[1] = {0.3, 0.4};
        factors[2] = {0.5, 0.6};
        model_->SetFactors(factors);
        
        // 创建测试数据
        features_.push_back(SparseFeature{0, 1.0});
        features_.push_back(SparseFeature{1, 2.0});
        features_.push_back(SparseFeature{2, 3.0});
        
        // 创建优化器
        optimizer_ = std::make_shared<SGDOptimizer>(0.1);
    }
    
    std::shared_ptr<Activation> activation_;
    std::shared_ptr<FMModel> model_;
    SparseFeatureVector features_;
    std::shared_ptr<Optimizer> optimizer_;
};

TEST_F(FMTest, Forward) {
    // 计算前向传播
    Float prediction = model_->Forward(features_);
    
    // 手动计算预期结果
    // 线性部分
    Float linear = 0.5 + 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0;
    
    // 交叉项部分
    Float interaction = 0.0;
    for (Int f = 0; f < 2; ++f) {
        Float sum_square = 0.0;
        Float square_sum = 0.0;
        
        // 计算交叉项
        for (const auto& feature : features_) {
            Int i = feature.index;
            Float value = feature.value;
            Float factor = model_->GetFactors().at(i).at(f);
            
            sum_square += (factor * value) * (factor * value);
            square_sum += factor * value;
        }
        square_sum = square_sum * square_sum;
        
        interaction += 0.5 * (square_sum - sum_square);
    }
    
    Float z = linear + interaction;
    Float expected = 1.0 / (1.0 + std::exp(-z));
    
    // 验证预测结果
    EXPECT_FLOAT_EQ(prediction, expected);
}

TEST_F(FMTest, Backward) {
    // 获取当前权重、因子和偏置
    auto original_weights = model_->GetWeights();
    auto original_factors = model_->GetFactors();
    Float original_bias = model_->GetBias();
    
    // 计算前向传播
    Float prediction = model_->Forward(features_);
    
    // 执行反向传播
    Float gradient = 0.5;  // 假设损失函数的梯度为0.5
    model_->Backward(features_, gradient, optimizer_);
    
    // 获取更新后的权重、因子和偏置
    auto updated_weights = model_->GetWeights();
    auto updated_factors = model_->GetFactors();
    Float updated_bias = model_->GetBias();
    
    // 验证偏置更新
    EXPECT_NE(updated_bias, original_bias);
    
    // 验证权重更新
    for (const auto& feature : features_) {
        Int index = feature.index;
        EXPECT_NE(updated_weights.at(index), original_weights.at(index));
    }
    
    // 验证因子更新
    for (const auto& feature : features_) {
        Int index = feature.index;
        for (Int f = 0; f < 2; ++f) {
            EXPECT_NE(updated_factors.at(index).at(f), original_factors.at(index).at(f));
        }
    }
}

TEST_F(FMTest, SaveAndLoad) {
    // 保存模型
    std::string model_file = "fm_model_test.dat";
    model_->Save(model_file);
    
    // 创建新模型并加载
    auto new_model = std::make_shared<FMModel>(3, 2, activation_);
    new_model->Load(model_file);
    
    // 验证加载的模型具有相同的权重、因子和偏置
    EXPECT_FLOAT_EQ(new_model->GetBias(), model_->GetBias());
    
    auto original_weights = model_->GetWeights();
    auto loaded_weights = new_model->GetWeights();
    
    for (const auto& pair : original_weights) {
        Int index = pair.first;
        Float weight = pair.second;
        
        EXPECT_FLOAT_EQ(loaded_weights.at(index), weight);
    }
    
    auto original_factors = model_->GetFactors();
    auto loaded_factors = new_model->GetFactors();
    
    for (const auto& pair : original_factors) {
        Int index = pair.first;
        const auto& factor = pair.second;
        
        for (Int f = 0; f < 2; ++f) {
            EXPECT_FLOAT_EQ(loaded_factors.at(index).at(f), factor.at(f));
        }
    }
    
    // 验证两个模型的预测结果相同
    Float pred1 = model_->Forward(features_);
    Float pred2 = new_model->Forward(features_);
    EXPECT_FLOAT_EQ(pred1, pred2);
    
    // 清理测试文件
    std::remove(model_file.c_str());
}

TEST_F(FMTest, Initialization) {
    // 创建新模型并初始化
    auto new_model = std::make_shared<FMModel>(3, 2, activation_);
    new_model->Init();
    
    // 验证初始化后的权重为空（懒加载）
    EXPECT_TRUE(new_model->GetWeights().empty());
    EXPECT_TRUE(new_model->GetFactors().empty());
    
    // 验证偏置被初始化为0
    EXPECT_FLOAT_EQ(new_model->GetBias(), 0.0);

}

} // namespace test
} // namespace simpleflow 