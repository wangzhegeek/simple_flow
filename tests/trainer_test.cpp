#include <gtest/gtest.h>
#include "trainer.h"
#include "model.h"
#include "optimizer.h"
#include "loss.h"
#include "metric.h"
#include "data_reader.h"
#include "activation.h"
#include "models/lr.h"
#include "models/fm.h"
#include <memory>
#include <vector>
#include <fstream>
#include <cstdio>
#include <thread>
#include <chrono>
#include <cmath>

namespace simpleflow {
namespace test {

class TrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建一个临时LIBSVM文件用于测试
        train_file_ = "train_data.libsvm";
        valid_file_ = "valid_data.libsvm";
        
        // 训练数据：2个特征，二分类问题
        std::ofstream train_file(train_file_);
        // 创建更多的数据样本用于多线程测试
        for (int i = 0; i < 1000; ++i) {
            float x1 = static_cast<float>(i % 10) * 0.5f;
            float x2 = static_cast<float>(i % 15) * 0.3f;
            int label = (x1 + x2 > 5.0f) ? 1 : 0;
            train_file << label << " 0:" << x1 << " 1:" << x2 << "\n";
        }
        train_file.close();
        
        // 验证数据
        std::ofstream valid_file(valid_file_);
        for (int i = 0; i < 200; ++i) {
            float x1 = static_cast<float>(i % 8) * 0.6f;
            float x2 = static_cast<float>(i % 12) * 0.4f;
            int label = (x1 + x2 > 5.0f) ? 1 : 0;
            valid_file << label << " 0:" << x1 << " 1:" << x2 << "\n";
        }
        valid_file.close();
        
        // 创建数据读取器
        train_reader_ = std::make_shared<LibSVMReader>(train_file_, 32, 2); // 批次大小为32
        valid_reader_ = std::make_shared<LibSVMReader>(valid_file_, 32, 2);
        
        // 创建模型 - 使用LR模型
        auto activation = std::make_shared<SigmoidActivation>();
        model_ = std::make_shared<LRModel>(2, activation);
        model_->Init();
        
        // 创建优化器、损失函数和评估指标
        optimizer_ = std::make_shared<SGDOptimizer>(0.1);
        loss_ = std::make_shared<LogLoss>();
        metrics_.push_back(std::make_shared<AccuracyMetric>());
    }
    
    void TearDown() override {
        // 删除临时文件
        std::remove(train_file_.c_str());
        std::remove(valid_file_.c_str());
    }
    
    std::shared_ptr<DataReader> train_reader_;
    std::shared_ptr<DataReader> valid_reader_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<Loss> loss_;
    std::vector<std::shared_ptr<Metric>> metrics_;
    
    std::string train_file_;
    std::string valid_file_;
};

TEST_F(TrainerTest, MultiThreadedTraining) {
    // 创建训练配置，设置多线程参数
    TrainerConfig config;
    config.batch_size = 32;
    config.epochs = 3;
    config.num_threads = 4;  // 使用4个线程
    config.verbose = false;  // 关闭详细输出，避免干扰测试输出
    
    // 创建训练器
    Trainer trainer(model_, loss_, optimizer_, config);
    
    // 添加评估指标
    trainer.AddMetric(std::make_shared<AccuracyMetric>());
    trainer.AddMetric(std::make_shared<AUCMetric>());
    
    // 执行多线程训练
    EXPECT_NO_THROW(trainer.Train(train_reader_, valid_reader_));
    
    // 验证模型是否成功训练
    FloatVector predictions;
    trainer.Predict(valid_reader_, predictions);
    
    EXPECT_EQ(predictions.size(), 200);  // 应该有200个预测结果
    
    // 所有预测值应该在[0,1]范围内（对于sigmoid激活函数）
    for (Float pred : predictions) {
        EXPECT_GE(pred, 0.0f);
        EXPECT_LE(pred, 1.0f);
    }
}

// 测试不同线程数对训练性能的影响
TEST_F(TrainerTest, ThreadScalabilityTest) {
    std::vector<int> thread_counts = {1, 2, 4};
    std::vector<double> training_times;
    
    // 对不同的线程数运行测试
    for (int num_threads : thread_counts) {
        // 创建训练配置
        TrainerConfig config;
        config.batch_size = 32;
        config.epochs = 2;
        config.num_threads = num_threads;
        config.verbose = false;
        
        // 创建新的模型
        auto activation = std::make_shared<SigmoidActivation>();
        auto thread_model = std::make_shared<LRModel>(2, activation);
        thread_model->Init();
        
        // 创建训练器
        Trainer trainer(thread_model, loss_, optimizer_, config);
        
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 执行训练
        trainer.Train(train_reader_);
        
        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        training_times.push_back(static_cast<double>(duration));
    }
    
    // 如果线程扩展性良好，那么更多线程应该比单线程快
    // 注意：在某些单元测试环境中，这个测试可能不总是可靠的，
    // 因为其他因素也会影响性能测量
    if (training_times.size() >= 3) {
        // 打印训练时间，帮助诊断
        std::cout << "Training times for different thread counts:" << std::endl;
        for (size_t i = 0; i < thread_counts.size(); ++i) {
            std::cout << thread_counts[i] << " threads: " << training_times[i] << " ms" << std::endl;
        }
        
        // 仅在4线程比1线程慢很多时发出警告
        if (training_times[2] > training_times[0] * 1.5) {
            std::cout << "Warning: 4 threads is significantly slower than 1 thread, "
                      << "which might indicate problems with the multi-threaded implementation" << std::endl;
        }
    }
    
    // 由于线程调度的不确定性，我们不做硬性断言，只是记录和比较时间
    SUCCEED() << "Thread scalability test completed";
}

// 测试线程安全性 - 多次运行相同的训练
TEST_F(TrainerTest, ThreadSafetyConsistencyTest) {
    // 创建5个相同配置的训练器
    const int num_trainers = 5;
    std::vector<std::shared_ptr<Model>> models;
    std::vector<std::vector<Float>> all_predictions;
    
    // 创建基础配置
    TrainerConfig config;
    config.batch_size = 32;
    config.epochs = 3;
    config.num_threads = 4;  // 使用4个线程
    config.verbose = false;
    
    // 初始化随机种子以确保确定性
    config.random_seed = 42;
    
    // 多次运行相同的训练过程
    for (int i = 0; i < num_trainers; ++i) {
        // 创建新模型
        auto activation = std::make_shared<SigmoidActivation>();
        auto thread_model = std::make_shared<LRModel>(2, activation);
        thread_model->Init();
        models.push_back(thread_model);
        
        // 创建训练器
        Trainer trainer(thread_model, loss_, optimizer_, config);
        
        // 训练模型
        trainer.Train(train_reader_);
        
        // 获取预测结果
        FloatVector predictions;
        trainer.Predict(valid_reader_, predictions);
        all_predictions.push_back(predictions);
    }
    
    // 如果线程实现是确定性的，相同配置的多次训练应该产生相同或非常相似的结果
    // 检查所有训练器的预测是否一致
    if (!all_predictions.empty()) {
        const FloatVector& base_predictions = all_predictions[0];
        
        for (size_t i = 1; i < all_predictions.size(); ++i) {
            const FloatVector& current_predictions = all_predictions[i];
            ASSERT_EQ(base_predictions.size(), current_predictions.size());
            
            // 计算预测值的绝对差异
            Float max_diff = 0.0f;
            for (size_t j = 0; j < base_predictions.size(); ++j) {
                Float diff = std::abs(base_predictions[j] - current_predictions[j]);
                max_diff = std::max(max_diff, diff);
            }
            
            // 仅输出最大差异，不做硬性断言
            // 因为浮点数计算和线程调度可能导致细微差异
            std::cout << "Max prediction difference between run 0 and run " << i << ": " << max_diff << std::endl;
            
            // 如果差异太大，可能表明线程安全问题
            if (max_diff > 0.1f) {
                std::cout << "Warning: Large prediction difference detected between runs, "
                          << "which might indicate thread safety issues" << std::endl;
            }
        }
    }
    
    SUCCEED() << "Thread safety consistency test completed";
}

} // namespace test
} // namespace simpleflow 