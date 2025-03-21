#include <gtest/gtest.h>
#include "simpleflow/activation.h"
#include <cmath>
#include <vector>
#include <memory>

namespace simpleflow {
namespace test {

class ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建各种激活函数
        identity_ = std::make_shared<IdentityActivation>();
        sigmoid_ = std::make_shared<SigmoidActivation>();
        relu_ = std::make_shared<ReLUActivation>();
        tanh_ = std::make_shared<TanhActivation>();
        
        // 测试数据
        inputs_ = {-2.0, -1.0, 0.0, 1.0, 2.0};
    }
    
    std::shared_ptr<Activation> identity_;
    std::shared_ptr<Activation> sigmoid_;
    std::shared_ptr<Activation> relu_;
    std::shared_ptr<Activation> tanh_;
    
    FloatVector inputs_;
};

TEST_F(ActivationTest, IdentityForward) {
    for (Float input : inputs_) {
        EXPECT_FLOAT_EQ(identity_->Forward(input), input);
    }
    
    FloatVector outputs;
    identity_->Forward(inputs_, outputs);
    
    ASSERT_EQ(outputs.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[i], inputs_[i]);
    }
}

TEST_F(ActivationTest, IdentityGradient) {
    for (Float input : inputs_) {
        Float output = identity_->Forward(input);
        EXPECT_FLOAT_EQ(identity_->Gradient(input, output), 1.0);
    }
    
    FloatVector outputs;
    identity_->Forward(inputs_, outputs);
    
    FloatVector gradients;
    identity_->Gradient(inputs_, outputs, gradients);
    
    ASSERT_EQ(gradients.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        EXPECT_FLOAT_EQ(gradients[i], 1.0);
    }
}

TEST_F(ActivationTest, SigmoidForward) {
    for (Float input : inputs_) {
        Float expected = 1.0 / (1.0 + std::exp(-input));
        EXPECT_FLOAT_EQ(sigmoid_->Forward(input), expected);
    }
    
    FloatVector outputs;
    sigmoid_->Forward(inputs_, outputs);
    
    ASSERT_EQ(outputs.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = 1.0 / (1.0 + std::exp(-inputs_[i]));
        EXPECT_FLOAT_EQ(outputs[i], expected);
    }
}

TEST_F(ActivationTest, SigmoidGradient) {
    for (Float input : inputs_) {
        Float output = sigmoid_->Forward(input);
        Float expected = output * (1.0 - output);
        EXPECT_FLOAT_EQ(sigmoid_->Gradient(input, output), expected);
    }
    
    FloatVector outputs;
    sigmoid_->Forward(inputs_, outputs);
    
    FloatVector gradients;
    sigmoid_->Gradient(inputs_, outputs, gradients);
    
    ASSERT_EQ(gradients.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = outputs[i] * (1.0 - outputs[i]);
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(ActivationTest, ReLUForward) {
    for (Float input : inputs_) {
        Float expected = std::max(0.0f, input);
        EXPECT_FLOAT_EQ(relu_->Forward(input), expected);
    }
    
    FloatVector outputs;
    relu_->Forward(inputs_, outputs);
    
    ASSERT_EQ(outputs.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = std::max(0.0f, inputs_[i]);
        EXPECT_FLOAT_EQ(outputs[i], expected);
    }
}

TEST_F(ActivationTest, ReLUGradient) {
    for (Float input : inputs_) {
        Float output = relu_->Forward(input);
        Float expected = (input > 0.0) ? 1.0 : 0.0;
        EXPECT_FLOAT_EQ(relu_->Gradient(input, output), expected);
    }
    
    FloatVector outputs;
    relu_->Forward(inputs_, outputs);
    
    FloatVector gradients;
    relu_->Gradient(inputs_, outputs, gradients);
    
    ASSERT_EQ(gradients.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = (inputs_[i] > 0.0) ? 1.0 : 0.0;
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(ActivationTest, TanhForward) {
    for (Float input : inputs_) {
        Float expected = std::tanh(input);
        EXPECT_FLOAT_EQ(tanh_->Forward(input), expected);
    }
    
    FloatVector outputs;
    tanh_->Forward(inputs_, outputs);
    
    ASSERT_EQ(outputs.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = std::tanh(inputs_[i]);
        EXPECT_FLOAT_EQ(outputs[i], expected);
    }
}

TEST_F(ActivationTest, TanhGradient) {
    for (Float input : inputs_) {
        Float output = tanh_->Forward(input);
        Float expected = 1.0 - output * output;
        EXPECT_FLOAT_EQ(tanh_->Gradient(input, output), expected);
    }
    
    FloatVector outputs;
    tanh_->Forward(inputs_, outputs);
    
    FloatVector gradients;
    tanh_->Gradient(inputs_, outputs, gradients);
    
    ASSERT_EQ(gradients.size(), inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Float expected = 1.0 - outputs[i] * outputs[i];
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(ActivationTest, Create) {
    std::shared_ptr<Activation> activation;
    
    activation = Activation::Create(ActivationType::Identity);
    EXPECT_FLOAT_EQ(activation->Forward(2.0), 2.0);
    
    activation = Activation::Create(ActivationType::Sigmoid);
    EXPECT_FLOAT_EQ(activation->Forward(0.0), 0.5);
    
    activation = Activation::Create(ActivationType::ReLU);
    EXPECT_FLOAT_EQ(activation->Forward(-1.0), 0.0);
    
    activation = Activation::Create(ActivationType::Tanh);
    EXPECT_FLOAT_EQ(activation->Forward(0.0), 0.0);
    
    EXPECT_THROW(Activation::Create(ActivationType::Unknown), std::runtime_error);
}

TEST_F(ActivationTest, ParseActivationType) {
    EXPECT_EQ(ParseActivationType("identity"), ActivationType::Identity);
    EXPECT_EQ(ParseActivationType("sigmoid"), ActivationType::Sigmoid);
    EXPECT_EQ(ParseActivationType("relu"), ActivationType::ReLU);
    EXPECT_EQ(ParseActivationType("tanh"), ActivationType::Tanh);
    EXPECT_EQ(ParseActivationType("unknown"), ActivationType::Unknown);
}

} // namespace test
} // namespace simpleflow 