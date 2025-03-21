#include <gtest/gtest.h>
#include "simpleflow/metric.h"
#include <cmath>
#include <memory>
#include <vector>

namespace simpleflow {
namespace test {

class MetricTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建二分类测试数据 (0/1标签格式)
        binary_predictions_ = {0.1, 0.3, 0.7, 0.9};
        binary_targets_ = {0.0, 0.0, 1.0, 1.0};
        
        // 创建二分类测试数据 (-1/+1标签格式)
        binary_predictions_pm_ = {0.1, 0.3, 0.7, 0.9};
        binary_targets_pm_ = {-1.0, -1.0, 1.0, 1.0};
    }
    
    FloatVector binary_predictions_;
    FloatVector binary_targets_;
    FloatVector binary_predictions_pm_;  // pm表示plus/minus，即-1/+1标签
    FloatVector binary_targets_pm_;
};

TEST_F(MetricTest, AccuracyCompute) {
    // 创建准确率指标
    auto accuracy = std::make_shared<AccuracyMetric>();
    
    // 添加二分类测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        accuracy->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // 手动计算准确率（阈值0.5）
    Float expected = 0.0;
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        bool pred_class = binary_predictions_[i] >= 0.5;
        bool true_class = binary_targets_[i] >= 0.5;
        
        if (pred_class == true_class) {
            expected += 1.0;
        }
    }
    expected /= binary_predictions_.size();
    
    // 验证准确率
    EXPECT_FLOAT_EQ(accuracy->Get(), expected);
}

TEST_F(MetricTest, AccuracyComputePlusMinus) {
    // 创建准确率指标
    auto accuracy = std::make_shared<AccuracyMetric>();
    
    // 添加-1/+1标签格式的测试数据
    for (size_t i = 0; i < binary_predictions_pm_.size(); ++i) {
        accuracy->Add(binary_predictions_pm_[i], binary_targets_pm_[i]);
    }
    
    // 手动计算准确率（阈值0.5）
    Float expected = 0.0;
    for (size_t i = 0; i < binary_predictions_pm_.size(); ++i) {
        bool pred_class = binary_predictions_pm_[i] >= 0.5;
        bool true_class = binary_targets_pm_[i] > 0;  // 正负标签的判定
        
        if (pred_class == true_class) {
            expected += 1.0;
        }
    }
    expected /= binary_predictions_pm_.size();
    
    // 验证准确率
    EXPECT_FLOAT_EQ(accuracy->Get(), expected);
}

TEST_F(MetricTest, AccuracyReset) {
    auto accuracy = std::make_shared<AccuracyMetric>();
    
    // 添加测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        accuracy->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // 验证有值
    EXPECT_NE(accuracy->Get(), 0.0);
    
    // 重置
    accuracy->Reset();
    
    // 验证重置后为零
    EXPECT_FLOAT_EQ(accuracy->Get(), 0.0);
}

TEST_F(MetricTest, AUCCompute) {
    // 创建AUC指标
    auto auc = std::make_shared<AUCMetric>();
    
    // 添加二分类测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        auc->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // AUC计算比较复杂，这里只验证值在合理范围内
    Float auc_value = auc->Get();
    EXPECT_GE(auc_value, 0.0);
    EXPECT_LE(auc_value, 1.0);
    
    // 对于我们的测试数据：
    // 排序后的预测值：[0.9, 0.7, 0.3, 0.1]
    // 对应的真实值：[1.0, 1.0, 0.0, 0.0]
    // 这是一个完美分类情况，所有正样本的预测值都大于负样本
    // 应该返回AUC=1.0
    EXPECT_FLOAT_EQ(auc_value, 1.0);
}

TEST_F(MetricTest, AUCComputePlusMinus) {
    // 创建AUC指标
    auto auc = std::make_shared<AUCMetric>();
    
    // 添加-1/+1标签格式的测试数据
    for (size_t i = 0; i < binary_predictions_pm_.size(); ++i) {
        auc->Add(binary_predictions_pm_[i], binary_targets_pm_[i]);
    }
    
    // AUC计算比较复杂，这里只验证值在合理范围内
    Float auc_value = auc->Get();
    EXPECT_GE(auc_value, 0.0);
    EXPECT_LE(auc_value, 1.0);
    
    // 对于我们的测试数据：
    // 排序后的预测值：[0.9, 0.7, 0.3, 0.1]
    // 对应的真实值：[1.0, 1.0, -1.0, -1.0]
    // 这是一个完美分类情况，所有正样本的预测值都大于负样本
    // 应该返回AUC=1.0
    EXPECT_FLOAT_EQ(auc_value, 1.0);
}

TEST_F(MetricTest, AUCReset) {
    auto auc = std::make_shared<AUCMetric>();
    
    // 添加测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        auc->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // 重置
    auc->Reset();
    
    // 验证重置后为默认值0.5（随机猜测的AUC）
    EXPECT_FLOAT_EQ(auc->Get(), 0.5);
}

TEST_F(MetricTest, LogLossCompute) {
    // 创建对数损失指标
    auto log_loss = std::make_shared<LogLossMetric>();
    
    // 添加二分类测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        log_loss->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // 手动计算对数损失
    Float expected = 0.0;
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        // 裁剪预测值，避免数值问题
        Float p = std::max(std::min(binary_predictions_[i], 1.0f - 1e-7f), 1e-7f);
        expected += -binary_targets_[i] * std::log(p) - (1.0f - binary_targets_[i]) * std::log(1.0f - p);
    }
    expected /= binary_predictions_.size();
    
    // 验证对数损失
    EXPECT_NEAR(log_loss->Get(), expected, 1e-6);
}

TEST_F(MetricTest, LogLossComputePlusMinus) {
    // 创建对数损失指标
    auto log_loss = std::make_shared<LogLossMetric>();
    
    // 添加-1/+1标签格式的测试数据
    for (size_t i = 0; i < binary_predictions_pm_.size(); ++i) {
        log_loss->Add(binary_predictions_pm_[i], binary_targets_pm_[i]);
    }
    
    // 手动计算对数损失，注意-1/+1标签的转换
    Float expected = 0.0;
    for (size_t i = 0; i < binary_predictions_pm_.size(); ++i) {
        // 裁剪预测值，避免数值问题
        Float p = std::max(std::min(binary_predictions_pm_[i], 1.0f - 1e-7f), 1e-7f);
        Float adjusted_target = binary_targets_pm_[i] > 0 ? 1.0f : 0.0f;
        expected += -adjusted_target * std::log(p) - (1.0f - adjusted_target) * std::log(1.0f - p);
    }
    expected /= binary_predictions_pm_.size();
    
    // 验证对数损失
    EXPECT_NEAR(log_loss->Get(), expected, 1e-6);
}

TEST_F(MetricTest, LogLossReset) {
    auto log_loss = std::make_shared<LogLossMetric>();
    
    // 添加测试数据
    for (size_t i = 0; i < binary_predictions_.size(); ++i) {
        log_loss->Add(binary_predictions_[i], binary_targets_[i]);
    }
    
    // 验证有值
    EXPECT_GT(log_loss->Get(), 0.0);
    
    // 重置
    log_loss->Reset();
    
    // 验证重置后为零
    EXPECT_FLOAT_EQ(log_loss->Get(), 0.0);
}

// 特殊情况：预测值和实际值的方向相反的情况
TEST_F(MetricTest, AUCInverseRelationship) {
    // 创建与标签方向相反的预测
    FloatVector inverse_predictions = {0.9, 0.7, 0.3, 0.1};
    FloatVector inverse_targets = {0.0, 0.0, 1.0, 1.0};  // 高分对应负样本，低分对应正样本
    
    auto auc = std::make_shared<AUCMetric>();
    
    // 添加反向关系的数据
    for (size_t i = 0; i < inverse_predictions.size(); ++i) {
        auc->Add(inverse_predictions[i], inverse_targets[i]);
    }
    
    // 由于我们的Gisette实现使用预测值排序并计算正样本排名，应该会得到0
    Float auc_value = auc->Get();
    EXPECT_FLOAT_EQ(auc_value, 0.0);
}

} // namespace test
} // namespace simpleflow 