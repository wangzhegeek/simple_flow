#include <gtest/gtest.h>
#include "model.h"
#include "activation.h"
#include "utils.h"
#include "optimizer.h"
#include "models/lr.h"
#include "models/fm.h"
#include <memory>
#include <vector>
#include <cmath>

namespace simpleflow {
namespace test {

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建一个LR模型
        auto activation = std::make_shared<SigmoidActivation>();
        lr_model_ = std::make_shared<LRModel>(3, activation);
        lr_model_->Init();
        
        // 创建一个FM模型
        fm_model_ = std::make_shared<FMModel>(3, 2, activation);
        fm_model_->Init();
        
        // 创建测试数据
        features_.push_back({0, 1.0});
        features_.push_back({1, 2.0});
        features_.push_back({2, 3.0});
    }
    
    std::shared_ptr<LRModel> lr_model_;
    std::shared_ptr<FMModel> fm_model_;
    SparseFeatureVector features_;
};

TEST_F(ModelTest, LRModelForward) {
    // 测试LR模型前向传播
    Float output = lr_model_->Forward(features_);
    EXPECT_GE(output, 0.0);
    EXPECT_LE(output, 1.0);
}

TEST_F(ModelTest, FMModelForward) {
    // 测试FM模型前向传播
    Float output = fm_model_->Forward(features_);
    EXPECT_GE(output, 0.0);
    EXPECT_LE(output, 1.0);
}

TEST_F(ModelTest, Create) {
    // 测试模型工厂方法
    std::shared_ptr<Model> model;
    
    // 创建LR模型
    model = Model::Create(ModelType::LR, 3);
    EXPECT_NE(model, nullptr);
    
    // 创建FM模型
    std::unordered_map<String, String> params = {{"embedding_size", "4"}};
    model = Model::Create(ModelType::FM, 3, params);
    EXPECT_NE(model, nullptr);
    
    // 未知模型类型
    EXPECT_THROW(Model::Create(ModelType::Unknown, 3), std::runtime_error);
}

TEST_F(ModelTest, ParseModelType) {
    EXPECT_EQ(ParseModelType("lr"), ModelType::LR);
    EXPECT_EQ(ParseModelType("LR"), ModelType::LR);
    EXPECT_EQ(ParseModelType("logistic_regression"), ModelType::LR);
    EXPECT_EQ(ParseModelType("LogisticRegression"), ModelType::LR);
    EXPECT_EQ(ParseModelType("fm"), ModelType::FM);
    EXPECT_EQ(ParseModelType("FM"), ModelType::FM);
    EXPECT_EQ(ParseModelType("factorization_machine"), ModelType::FM);
    EXPECT_EQ(ParseModelType("FactorizationMachine"), ModelType::FM);
    EXPECT_EQ(ParseModelType("unknown"), ModelType::Unknown);
}

} // namespace test
} // namespace simpleflow 