#include <gtest/gtest.h>
#include "optimizer.h"
#include <cmath>
#include <memory>
#include <vector>
#include <unordered_map>

namespace simpleflow {
namespace test {

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试数据
        weights_ = {1.0, 2.0, 3.0, 4.0, 5.0};
        gradients_ = {0.1, 0.2, 0.3, 0.4, 0.5};
        
        // 创建优化器
        sgd_ = std::make_shared<SGDOptimizer>(0.1);
        adagrad_ = std::make_shared<AdagradOptimizer>(0.1, 1e-8);
        rmsprop_ = std::make_shared<RMSPropOptimizer>(0.1, 0.0, 0.9, 1e-8);
        adam_ = std::make_shared<AdamOptimizer>(0.1, 0.0, 0.9, 0.999, 1e-8);
        
        // 使用共享权重进行初始化
        weights_sgd_ = weights_;
        weights_adagrad_ = weights_;
        weights_rmsprop_ = weights_;
        weights_adam_ = weights_;
    }
    
    FloatVector weights_;
    FloatVector gradients_;
    
    std::shared_ptr<Optimizer> sgd_;
    std::shared_ptr<Optimizer> adagrad_;
    std::shared_ptr<Optimizer> rmsprop_;
    std::shared_ptr<Optimizer> adam_;
    
    FloatVector weights_sgd_;
    FloatVector weights_adagrad_;
    FloatVector weights_rmsprop_;
    FloatVector weights_adam_;
};

TEST_F(OptimizerTest, SGDUpdate) {
    // 手动计算SGD更新
    FloatVector expected = weights_;
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] -= 0.1 * gradients_[i];
    }
    
    // 使用优化器更新
    sgd_->Update(weights_sgd_, gradients_);
    
    // 验证更新结果
    ASSERT_EQ(weights_sgd_.size(), expected.size());
    for (size_t i = 0; i < weights_sgd_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_sgd_[i], expected[i]);
    }
}

TEST_F(OptimizerTest, AdaGradUpdate) {
    // 手动计算AdaGrad更新
    FloatVector expected = weights_;
    FloatVector cache(expected.size(), 0.0);
    
    // 第一次更新
    for (size_t i = 0; i < expected.size(); ++i) {
        cache[i] += gradients_[i] * gradients_[i];
        expected[i] -= 0.1 * gradients_[i] / (std::sqrt(cache[i]) + 1e-8);
    }
    
    // 使用优化器更新
    adagrad_->Update(weights_adagrad_, gradients_);
    
    // 验证更新结果
    ASSERT_EQ(weights_adagrad_.size(), expected.size());
    for (size_t i = 0; i < weights_adagrad_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_adagrad_[i], expected[i]);
    }
    
    // 第二次更新
    for (size_t i = 0; i < expected.size(); ++i) {
        cache[i] += gradients_[i] * gradients_[i];
        expected[i] -= 0.1 * gradients_[i] / (std::sqrt(cache[i]) + 1e-8);
    }
    
    adagrad_->Update(weights_adagrad_, gradients_);
    
    // 验证第二次更新结果
    for (size_t i = 0; i < weights_adagrad_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_adagrad_[i], expected[i]);
    }
}

TEST_F(OptimizerTest, RMSPropUpdate) {
    // 检查实际使用的RMSProp优化器参数
    Float learning_rate = 0.1;
    Float l2_reg = 0.0;
    Float decay_rate = 0.9;
    Float epsilon = 1e-8;
    
    // 创建一个缓存数组，与优化器相同的方式存储和更新缓存
    std::unordered_map<Int, Float> cache;
    FloatVector expected = weights_;
    
    // 第一次更新
    for (size_t i = 0; i < expected.size(); ++i) {
        // 应用L2正则化
        Float grad = gradients_[i] + l2_reg * expected[i];
        
        // 更新缓存，与RMSPropOptimizer::Update实现保持一致
        cache[i] = decay_rate * cache[i] + (1 - decay_rate) * grad * grad;
        
        // 更新参数
        expected[i] -= learning_rate * grad / (std::sqrt(cache[i]) + epsilon);
    }
    
    // 使用优化器更新
    rmsprop_->Update(weights_rmsprop_, gradients_);
    
    // 验证更新结果
    ASSERT_EQ(weights_rmsprop_.size(), expected.size());
    for (size_t i = 0; i < weights_rmsprop_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_rmsprop_[i], expected[i]);
    }
    
    // 第二次更新
    for (size_t i = 0; i < expected.size(); ++i) {
        // 应用L2正则化
        Float grad = gradients_[i] + l2_reg * expected[i];
        
        // 更新缓存
        cache[i] = decay_rate * cache[i] + (1 - decay_rate) * grad * grad;
        
        // 更新参数
        expected[i] -= learning_rate * grad / (std::sqrt(cache[i]) + epsilon);
    }
    
    // 使用优化器更新第二次
    rmsprop_->Update(weights_rmsprop_, gradients_);
    
    // 验证第二次更新结果
    for (size_t i = 0; i < weights_rmsprop_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_rmsprop_[i], expected[i]);
    }
}

TEST_F(OptimizerTest, AdamUpdateDeepVerification) {
    // 准备副本权重和理论权重
    FloatVector weights_copy = weights_;
    FloatVector weights_theory = weights_;
    
    // 创建一个新的Adam优化器与测试中使用的参数完全一致
    std::shared_ptr<Optimizer> adam_copy = std::make_shared<AdamOptimizer>(0.1, 0.0, 0.9, 0.999, 1e-8);
    
    // 第一步：分别使用两个优化器更新权重
    adam_->Update(weights_adam_, gradients_);
    adam_copy->Update(weights_copy, gradients_);
    
    // 验证两个优化器实例行为一致（应该完全一样）
    for (size_t i = 0; i < weights_adam_.size(); ++i) {
        EXPECT_FLOAT_EQ(weights_adam_[i], weights_copy[i]);
    }
    
    // 第二步：理论计算权重更新
    // Adam算法参数
    Float learning_rate = 0.1;
    Float beta1 = 0.9;
    Float beta2 = 0.999;
    Float epsilon = 1e-8;
    
    // 手动模拟Adam的第一次更新
    std::vector<Float> m(weights_theory.size(), 0.0);
    std::vector<Float> v(weights_theory.size(), 0.0);
    int t = 0; // 初始化为0，与实际实现一致
    
    // 每个参数向量的更新都会导致t增加一次
    ++t;
    
    for (size_t i = 0; i < weights_theory.size(); ++i) {
        // 先应用L2正则化 (本测试中L2_reg为0)
        Float grad = gradients_[i];
        
        // 更新动量
        m[i] = beta1 * 0.0 + (1.0 - beta1) * grad;
        // 更新二阶矩
        v[i] = beta2 * 0.0 + (1.0 - beta2) * grad * grad;
        
        // 计算偏差修正
        Float m_hat = m[i] / (1.0 - std::pow(beta1, t));
        Float v_hat = v[i] / (1.0 - std::pow(beta2, t));
        
        // 应用更新
        weights_theory[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    
    // 对比理论值与实际值，使用宽松容差
    for (size_t i = 0; i < weights_adam_.size(); ++i) {
        EXPECT_NEAR(weights_adam_[i], weights_theory[i], 1e-4) 
            << "i=" << i
            << " weights_[i]=" << weights_[i]
            << " gradients_[i]=" << gradients_[i]
            << " m[i]=" << m[i]
            << " v[i]=" << v[i]
            << " theory=" << weights_theory[i]
            << " actual=" << weights_adam_[i];
    }
    
    // 验证第二次更新 - 使用同样的两个优化器
    adam_->Update(weights_adam_, gradients_);
    adam_copy->Update(weights_copy, gradients_);
    
    // 再次计算理论值进行验证
    ++t; // 第二次更新增加t
    
    for (size_t i = 0; i < weights_theory.size(); ++i) {
        Float grad = gradients_[i];
        
        // 更新动量 (使用上一次计算的m和v值)
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
        // 更新二阶矩
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
        
        // 计算偏差修正
        Float m_hat = m[i] / (1.0 - std::pow(beta1, t));
        Float v_hat = v[i] / (1.0 - std::pow(beta2, t));
        
        // 应用更新
        weights_theory[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    
    // 对比第二次更新后的理论值与实际值
    for (size_t i = 0; i < weights_adam_.size(); ++i) {
        EXPECT_NEAR(weights_adam_[i], weights_theory[i], 1e-4)
            << "Second update - i=" << i
            << " weights_[i]=" << weights_[i]
            << " gradients_[i]=" << gradients_[i]
            << " m[i]=" << m[i]
            << " v[i]=" << v[i]
            << " theory=" << weights_theory[i]
            << " actual=" << weights_adam_[i];
    }
}

TEST_F(OptimizerTest, AdamUpdate) {
    // Adam优化器和其他优化器的一个主要区别是它会记住过去的梯度信息，并为每个参数适应不同的学习率
    // 我们关注验证一些基本行为特性，而不是精确的数值
    
    // 准备特殊的梯度序列，有显著差异的梯度
    FloatVector special_gradients = {0.01, 1.0, 0.01};  // 中间梯度比两边大100倍
    FloatVector weights = {1.0, 1.0, 1.0};
    
    // 创建Adam优化器
    auto special_adam = std::make_shared<AdamOptimizer>(0.1);
    
    // 连续更新几轮
    const int num_updates = 5;
    for (int i = 0; i < num_updates; ++i) {
        special_adam->Update(weights, special_gradients);
    }
    
    // 验证更新后的权重：梯度大的参数权重应该减少更多
    EXPECT_LT(weights[1], weights[0]) << "梯度大的参数权重应该减少更多";
    EXPECT_LT(weights[1], weights[2]) << "梯度大的参数权重应该减少更多";
    
    // 验证基本优化行为
    FloatVector original_weights = weights_;
    
    // 第一次更新
    adam_->Update(weights_adam_, gradients_);
    
    // 验证更新发生且方向正确
    for (size_t i = 0; i < weights_adam_.size(); ++i) {
        // 由于梯度是正的，权重必须减小
        EXPECT_LT(weights_adam_[i], original_weights[i]) 
            << "梯度为正时，权重应该减小，但权重增加了";
    }
    
    // 保存第一次更新后的权重
    FloatVector updated_once = weights_adam_;
    
    // 第二次更新
    adam_->Update(weights_adam_, gradients_);
    
    // 验证第二次更新也有效
    for (size_t i = 0; i < weights_adam_.size(); ++i) {
        EXPECT_LT(weights_adam_[i], updated_once[i])
            << "第二次更新后权重应继续减小";
    }
}

TEST_F(OptimizerTest, Create) {
    // 测试创建各种优化器
    std::shared_ptr<Optimizer> optimizer;
    
    // SGD优化器
    optimizer = Optimizer::Create(OptimizerType::SGD, 0.1);
    FloatVector weights_copy = weights_;
    optimizer->Update(weights_copy, gradients_);
    EXPECT_FLOAT_EQ(weights_copy[0], weights_[0] - 0.1 * gradients_[0]);
    
    // AdaGrad优化器
    optimizer = Optimizer::Create(OptimizerType::Adagrad, 0.1);
    weights_copy = weights_;
    optimizer->Update(weights_copy, gradients_);
    EXPECT_NE(weights_copy[0], weights_[0]);
    
    // RMSProp优化器
    optimizer = Optimizer::Create(OptimizerType::RMSProp, 0.1);
    weights_copy = weights_;
    optimizer->Update(weights_copy, gradients_);
    EXPECT_NE(weights_copy[0], weights_[0]);
    
    // Adam优化器
    optimizer = Optimizer::Create(OptimizerType::Adam, 0.1);
    weights_copy = weights_;
    optimizer->Update(weights_copy, gradients_);
    EXPECT_NE(weights_copy[0], weights_[0]);
    
    // 未知优化器类型
    EXPECT_THROW(Optimizer::Create(OptimizerType::Unknown, 0.1), std::runtime_error);
}

TEST_F(OptimizerTest, ParseOptimizerType) {
    EXPECT_EQ(ParseOptimizerType("sgd"), OptimizerType::SGD);
    EXPECT_EQ(ParseOptimizerType("SGD"), OptimizerType::SGD);
    EXPECT_EQ(ParseOptimizerType("adagrad"), OptimizerType::Adagrad);
    EXPECT_EQ(ParseOptimizerType("AdaGrad"), OptimizerType::Adagrad);
    EXPECT_EQ(ParseOptimizerType("rmsprop"), OptimizerType::RMSProp);
    EXPECT_EQ(ParseOptimizerType("RMSProp"), OptimizerType::RMSProp);
    EXPECT_EQ(ParseOptimizerType("adam"), OptimizerType::Adam);
    EXPECT_EQ(ParseOptimizerType("Adam"), OptimizerType::Adam);
    EXPECT_EQ(ParseOptimizerType("unknown"), OptimizerType::Unknown);
}

} // namespace test
} // namespace simpleflow 