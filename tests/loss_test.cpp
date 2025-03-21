#include <gtest/gtest.h>
#include "simpleflow/loss.h"
#include <cmath>
#include <vector>
#include <memory>

namespace simpleflow {
namespace test {

class LossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建各种损失函数
        mse_ = std::make_shared<MSELoss>();
        logloss_ = std::make_shared<LogLoss>();
        hinge_ = std::make_shared<HingeLoss>();
        
        // 测试数据 (0/1标签)
        predictions_ = {0.1, 0.3, 0.5, 0.7, 0.9};
        targets_ = {0.0, 0.0, 1.0, 1.0, 1.0};
        
        // 测试数据 (-1/+1标签)
        predictions_pm_ = {0.1, 0.3, 0.5, 0.7, 0.9};
        targets_pm_ = {-1.0, -1.0, 1.0, 1.0, 1.0};
    }
    
    std::shared_ptr<Loss> mse_;
    std::shared_ptr<Loss> logloss_;
    std::shared_ptr<Loss> hinge_;
    
    FloatVector predictions_;
    FloatVector targets_;
    FloatVector predictions_pm_;  // pm表示plus/minus，即-1/+1标签
    FloatVector targets_pm_;
};

TEST_F(LossTest, MSECompute) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        Float expected = 0.5 * (pred - target) * (pred - target);
        EXPECT_FLOAT_EQ(mse_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float diff = predictions_[i] - targets_[i];
        expected_avg += 0.5 * diff * diff;
    }
    expected_avg /= predictions_.size();
    
    EXPECT_FLOAT_EQ(mse_->Compute(predictions_, targets_), expected_avg);
}

TEST_F(LossTest, MSEComputePlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        Float expected = 0.5 * (pred - target) * (pred - target);
        EXPECT_FLOAT_EQ(mse_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float diff = predictions_pm_[i] - targets_pm_[i];
        expected_avg += 0.5 * diff * diff;
    }
    expected_avg /= predictions_pm_.size();
    
    EXPECT_FLOAT_EQ(mse_->Compute(predictions_pm_, targets_pm_), expected_avg);
}

TEST_F(LossTest, MSEGradient) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        Float expected = pred - target;
        EXPECT_FLOAT_EQ(mse_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    mse_->Gradient(predictions_, targets_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_.size());
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float expected = predictions_[i] - targets_[i];
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, MSEGradientPlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        Float expected = pred - target;
        EXPECT_FLOAT_EQ(mse_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    mse_->Gradient(predictions_pm_, targets_pm_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_pm_.size());
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float expected = predictions_pm_[i] - targets_pm_[i];
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, LogLossCompute) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        // 裁剪预测值以避免数值问题
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = -target * std::log(p) - (1.0f - target) * std::log(1.0f - p);
        
        EXPECT_FLOAT_EQ(logloss_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        expected_avg += -target * std::log(p) - (1.0f - target) * std::log(1.0f - p);
    }
    expected_avg /= predictions_.size();
    
    EXPECT_FLOAT_EQ(logloss_->Compute(predictions_, targets_), expected_avg);
}

TEST_F(LossTest, LogLossComputePlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，我们需要转换为0/1格式
        Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
        
        // 裁剪预测值以避免数值问题
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = -adjusted_target * std::log(p) - (1.0f - adjusted_target) * std::log(1.0f - p);
        
        EXPECT_FLOAT_EQ(logloss_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，我们需要转换为0/1格式
        Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
        
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        expected_avg += -adjusted_target * std::log(p) - (1.0f - adjusted_target) * std::log(1.0f - p);
    }
    expected_avg /= predictions_pm_.size();
    
    EXPECT_FLOAT_EQ(logloss_->Compute(predictions_pm_, targets_pm_), expected_avg);
}

TEST_F(LossTest, LogLossGradient) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        // 裁剪预测值以避免数值问题
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = (p - target) / (p * (1.0f - p));
        
        EXPECT_FLOAT_EQ(logloss_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    logloss_->Gradient(predictions_, targets_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_.size());
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = (p - target) / (p * (1.0f - p));
        
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, LogLossGradientPlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，我们需要转换为0/1格式
        Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
        
        // 裁剪预测值以避免数值问题
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = (p - adjusted_target) / (p * (1.0f - p));
        
        EXPECT_FLOAT_EQ(logloss_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    logloss_->Gradient(predictions_pm_, targets_pm_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_pm_.size());
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，我们需要转换为0/1格式
        Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
        
        Float p = std::max(std::min(pred, 1.0f - 1e-7f), 1e-7f);
        Float expected = (p - adjusted_target) / (p * (1.0f - p));
        
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, HingeLossCompute) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        // 将标签从[0,1]映射到[-1,1]
        Float t = 2.0f * target - 1.0f;
        Float expected = std::max(0.0f, 1.0f - t * pred);
        
        EXPECT_FLOAT_EQ(hinge_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        Float t = 2.0f * target - 1.0f;
        expected_avg += std::max(0.0f, 1.0f - t * pred);
    }
    expected_avg /= predictions_.size();
    
    EXPECT_FLOAT_EQ(hinge_->Compute(predictions_, targets_), expected_avg);
}

TEST_F(LossTest, HingeLossComputePlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，直接使用
        Float t = (target > 0) ? 1.0f : -1.0f;
        Float expected = std::max(0.0f, 1.0f - t * pred);
        
        EXPECT_FLOAT_EQ(hinge_->Compute(pred, target), expected);
    }
    
    Float expected_avg = 0.0;
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，直接使用
        Float t = (target > 0) ? 1.0f : -1.0f;
        expected_avg += std::max(0.0f, 1.0f - t * pred);
    }
    expected_avg /= predictions_pm_.size();
    
    EXPECT_FLOAT_EQ(hinge_->Compute(predictions_pm_, targets_pm_), expected_avg);
}

TEST_F(LossTest, HingeLossGradient) {
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        // 将标签从[0,1]映射到[-1,1]
        Float t = 2.0f * target - 1.0f;
        Float expected = (t * pred < 1.0f) ? -t : 0.0f;
        
        EXPECT_FLOAT_EQ(hinge_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    hinge_->Gradient(predictions_, targets_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_.size());
    for (size_t i = 0; i < predictions_.size(); ++i) {
        Float pred = predictions_[i];
        Float target = targets_[i];
        
        Float t = 2.0f * target - 1.0f;
        Float expected = (t * pred < 1.0f) ? -t : 0.0f;
        
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, HingeLossGradientPlusMinus) {
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，直接使用
        Float t = (target > 0) ? 1.0f : -1.0f;
        Float expected = (t * pred < 1.0f) ? -t : 0.0f;
        
        EXPECT_FLOAT_EQ(hinge_->Gradient(pred, target), expected);
    }
    
    FloatVector gradients;
    hinge_->Gradient(predictions_pm_, targets_pm_, gradients);
    
    ASSERT_EQ(gradients.size(), predictions_pm_.size());
    for (size_t i = 0; i < predictions_pm_.size(); ++i) {
        Float pred = predictions_pm_[i];
        Float target = targets_pm_[i];
        
        // 对于-1/+1标签，直接使用
        Float t = (target > 0) ? 1.0f : -1.0f;
        Float expected = (t * pred < 1.0f) ? -t : 0.0f;
        
        EXPECT_FLOAT_EQ(gradients[i], expected);
    }
}

TEST_F(LossTest, Create) {
    std::shared_ptr<Loss> loss;
    
    loss = Loss::Create(LossType::MSE);
    EXPECT_FLOAT_EQ(loss->Compute(1.0, 0.0), 0.5);
    
    loss = Loss::Create(LossType::LogLoss);
    EXPECT_FLOAT_EQ(loss->Compute(0.5, 0.0), -std::log(0.5));
    
    loss = Loss::Create(LossType::Hinge);
    EXPECT_FLOAT_EQ(loss->Compute(0.5, 0.0), 1.5);
    
    EXPECT_THROW(Loss::Create(LossType::Unknown), std::runtime_error);
}

TEST_F(LossTest, ParseLossType) {
    EXPECT_EQ(ParseLossType("mse"), LossType::MSE);
    EXPECT_EQ(ParseLossType("logloss"), LossType::LogLoss);
    EXPECT_EQ(ParseLossType("hinge"), LossType::Hinge);
    EXPECT_EQ(ParseLossType("unknown"), LossType::Unknown);
}

} // namespace test
} // namespace simpleflow 