#include "metric.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace simpleflow {

void Metric::Add(const FloatVector& predictions, const FloatVector& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets size mismatch");
    }
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        Add(predictions[i], targets[i]);
    }
}

// 准确率指标
AccuracyMetric::AccuracyMetric(Float threshold)
    : correct_count_(0), total_count_(0), threshold_(threshold) {
}

void AccuracyMetric::Add(Float prediction, Float target) {
    bool pred_class = prediction >= threshold_;
    
    // 对于Gisette数据集，标签为-1/+1
    // 我们将大于0的标签视为正类
    bool true_class = target > 0;
    
    if (pred_class == true_class) {
        ++correct_count_;
    }
    
    ++total_count_;
}

Float AccuracyMetric::Get() const {
    return total_count_ > 0 ? static_cast<Float>(correct_count_) / total_count_ : 0.0;
}

void AccuracyMetric::Reset() {
    correct_count_ = 0;
    total_count_ = 0;
}

// AUC指标
AUCMetric::AUCMetric() {
    Reset();
}

void AUCMetric::Add(Float prediction, Float target) {
    predictions_.emplace_back(prediction, target);
}

void AUCMetric::Add(const FloatVector& predictions, const FloatVector& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets size mismatch");
    }
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Gisette数据集中标签可能是-1和1，而不是0和1
        Int label = (targets[i] > 0) ? 1 : 0;
        predictions_.emplace_back(predictions[i], label);
    }
}

Float AUCMetric::Get() const {
    return CalculateAUC();
}

void AUCMetric::Reset() {
    predictions_.clear();
}

Float AUCMetric::CalculateAUC() const {
    if (predictions_.empty()) {
        return 0.5;  // 默认值为随机猜测的AUC
    }
    
    // 复制预测结果，避免修改原始数据
    auto sorted_predictions = predictions_;
    
    // 按预测值降序排序
    std::sort(sorted_predictions.begin(), sorted_predictions.end(),
             [](const std::pair<Float, Int>& a, const std::pair<Float, Int>& b) {
                 return a.first > b.first;
             });
    
    // 计算正样本和负样本数量
    Int num_positives = 0;
    Int num_negatives = 0;
    
    for (const auto& p : sorted_predictions) {
        if (p.second > 0) {
            ++num_positives;
        } else {
            ++num_negatives;
        }
    }
    
    if (num_positives == 0 || num_negatives == 0) {
        return 0.5;  // 只有一个类别，无法计算AUC
    }
    
    // 使用和sparse_lr_multi_thread.cpp相同的方法计算AUC
    Float auc = 0.0;
    Int pos_count = 0;
    
    for (const auto& p : sorted_predictions) {
        if (p.second > 0) {
            // 正样本
            pos_count++;
        } else {
            // 负样本，增加AUC值
            auc += pos_count;
        }
    }
    
    // 使用安全的方式归一化AUC，避免整数溢出
    // 将整数转换为double类型进行计算
    double pos_d = static_cast<double>(num_positives);
    double neg_d = static_cast<double>(num_negatives);
    double denominator = pos_d * neg_d;
    // 计算最终AUC值
    Float final_auc = static_cast<Float>(static_cast<double>(auc) / denominator);
    
    return final_auc;
}

// 对数损失指标
LogLossMetric::LogLossMetric() : sum_loss_(0.0), count_(0) {
}

void LogLossMetric::Add(Float prediction, Float target) {
    // 处理可能的-1/+1标签
    Float adjusted_target = (target > 0) ? 1.0f : 0.0f;
    
    // 裁剪预测值，避免数值问题
    Float p = std::max(std::min(prediction, 1.0f - 1e-7f), 1e-7f);
    sum_loss_ += -adjusted_target * std::log(p) - (1.0f - adjusted_target) * std::log(1.0f - p);
    ++count_;
}

Float LogLossMetric::Get() const {
    return count_ > 0 ? sum_loss_ / count_ : 0.0;
}

void LogLossMetric::Reset() {
    sum_loss_ = 0.0;
    count_ = 0;
}

} // namespace simpleflow 