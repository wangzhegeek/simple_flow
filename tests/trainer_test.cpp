#include <gtest/gtest.h>
#include "simpleflow/trainer.h"
#include "simpleflow/model.h"
#include "simpleflow/optimizer.h"
#include "simpleflow/loss.h"
#include "simpleflow/metric.h"
#include "simpleflow/data_reader.h"
#include <memory>
#include <vector>
#include <fstream>
#include <cstdio>

namespace simpleflow {
namespace test {

class TrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建一个临时CSV文件用于测试
        train_file_ = "train_data.csv";
        valid_file_ = "valid_data.csv";
        
        // 训练数据：2个特征，二分类问题
        std::ofstream train_file(train_file_);
        train_file << "1.0,2.0,0\n"
                  << "2.0,3.0,0\n"
                  << "3.0,4.0,1\n"
                  << "4.0,5.0,1\n";
        train_file.close();
        
        // 验证数据
        std::ofstream valid_file(valid_file_);
        valid_file << "1.5,2.5,0\n"
                  << "3.5,4.5,1\n";
        valid_file.close();
        
        // 创建数据读取器
        train_reader_ = std::make_shared<CSVReader>(train_file_, 2, std::vector<size_t>{0, 1});
        valid_reader_ = std::make_shared<CSVReader>(valid_file_, 2, std::vector<size_t>{0, 1});
        
        // 创建模型 - 使用逻辑回归
        model_ = std::make_shared<LogisticRegression>(2);
        
        // 创建优化器、损失函数和评估指标
        optimizer_ = std::make_shared<SGDOptimizer>(0.1);
        loss_ = std::make_shared<LogLoss>();
        metrics_ = {std::make_shared<AccuracyMetric>()};
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

TEST_F(TrainerTest, TrainerConstruction) {
    // 测试训练器构造
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 验证训练器属性
    EXPECT_EQ(trainer.GetModel(), model_);
    EXPECT_EQ(trainer.GetOptimizer(), optimizer_);
    EXPECT_EQ(trainer.GetLoss(), loss_);
    EXPECT_EQ(trainer.GetMetrics().size(), metrics_.size());
}

TEST_F(TrainerTest, TrainOneEpoch) {
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 训练一个周期
    FloatVector losses = trainer.TrainOneEpoch(train_reader_, 2);
    
    // 验证损失向量的大小 (应该有2个批次，每个批次2个样本)
    EXPECT_EQ(losses.size(), 2);
    
    // 所有损失应该是有限的（非NaN，非无穷大）
    for (Float loss : losses) {
        EXPECT_TRUE(std::isfinite(loss));
    }
}

TEST_F(TrainerTest, Evaluate) {
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 进行一些训练
    trainer.TrainOneEpoch(train_reader_, 2);
    
    // 评估模型
    Float avg_loss;
    std::vector<Float> metric_values;
    trainer.Evaluate(valid_reader_, avg_loss, metric_values);
    
    // 验证评估结果
    EXPECT_TRUE(std::isfinite(avg_loss));
    EXPECT_EQ(metric_values.size(), metrics_.size());
    for (Float metric : metric_values) {
        EXPECT_TRUE(std::isfinite(metric));
    }
}

TEST_F(TrainerTest, Train) {
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 训练多个周期
    size_t num_epochs = 2;
    size_t batch_size = 2;
    size_t print_every = 1;
    
    // 进行训练
    EXPECT_NO_THROW(trainer.Train(train_reader_, valid_reader_, 
                                num_epochs, batch_size, print_every));
}

TEST_F(TrainerTest, SaveAndLoadModel) {
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 训练模型
    trainer.TrainOneEpoch(train_reader_, 2);
    
    // 保存模型
    std::string model_file = "test_model.dat";
    EXPECT_NO_THROW(trainer.SaveModel(model_file));
    
    // 创建一个新模型和训练器
    auto new_model = std::make_shared<LogisticRegression>(2);
    Trainer new_trainer(new_model, optimizer_, loss_, metrics_);
    
    // 加载模型
    EXPECT_NO_THROW(new_trainer.LoadModel(model_file));
    
    // 两个模型应该得到相同的预测
    FloatMatrix features;
    FloatVector labels;
    valid_reader_->GetBatch(0, 2, features, labels);
    
    FloatVector predictions1, predictions2;
    model_->Forward(features, predictions1);
    new_model->Forward(features, predictions2);
    
    ASSERT_EQ(predictions1.size(), predictions2.size());
    for (size_t i = 0; i < predictions1.size(); ++i) {
        EXPECT_FLOAT_EQ(predictions1[i], predictions2[i]);
    }
    
    // 清理测试文件
    std::remove(model_file.c_str());
}

TEST_F(TrainerTest, SetLearningRate) {
    Trainer trainer(model_, optimizer_, loss_, metrics_);
    
    // 设置新的学习率
    Float new_lr = 0.05;
    trainer.SetLearningRate(new_lr);
    
    // 在这个测试中，我们只验证SetLearningRate方法不会抛出异常
    // 我们无法直接验证优化器内部的学习率已更改，因为没有getter方法
    EXPECT_NO_THROW(trainer.TrainOneEpoch(train_reader_, 2));
}

} // namespace test
} // namespace simpleflow 