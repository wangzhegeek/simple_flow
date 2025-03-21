#ifndef SIMPLEFLOW_TRAINER_H
#define SIMPLEFLOW_TRAINER_H

#include "types.h"
#include "model.h"
#include "data_reader.h"
#include "loss.h"
#include "optimizer.h"
#include "metric.h"
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <functional>
#include <limits>
#include <iostream>

namespace simpleflow {

// 训练配置
struct TrainerConfig {
    Int batch_size = 128;
    Int epochs = 10;
    Int num_threads = 1;
    bool verbose = true;
    String model_save_path = "";
    Int random_seed = 42;
    bool shuffle = true;
    Int lr_decay_epochs = 3;        // 学习率衰减的轮数
    Float lr_decay_factor = 0.5f;   // 学习率衰减的因子
    Float min_learning_rate = 1e-6f; // 最小学习率
    Int log_interval = 100;         // 日志打印频率（每多少批次打印一次）
    
    TrainerConfig() = default;
};

// 训练器类
class Trainer {
public:
    Trainer(std::shared_ptr<Model> model, 
            std::shared_ptr<Loss> loss,
            std::shared_ptr<Optimizer> optimizer,
            const TrainerConfig& config = TrainerConfig());
    
    ~Trainer();
    
    // 训练模型
    void Train(std::shared_ptr<DataReader> train_reader, std::shared_ptr<DataReader> valid_reader = nullptr);
    
    // 评估模型
    void Evaluate(std::shared_ptr<DataReader> data_reader, const std::vector<std::shared_ptr<Metric>>& metrics);
    
    // 预测
    void Predict(std::shared_ptr<DataReader> data_reader, FloatVector& predictions);
    
    // 添加评估指标
    void AddMetric(std::shared_ptr<Metric> metric);
    
    // 设置训练回调函数
    void SetCallback(std::function<void(Int epoch, Float train_loss, const std::vector<Float>& metric_values)> callback);
    
    // 设置随机种子
    void SetRandomSeed(Int seed);
    
    // 设置是否打乱数据
    void SetShuffle(bool shuffle);
    
    // 设置批次大小
    void SetBatchSize(Int batch_size);
    
    // 设置训练轮数
    void SetEpochNum(Int epoch_num);
    
    // 设置详细程度
    void SetVerbose(Int verbose);
    
    // 设置学习率衰减轮数
    void SetLRDecayEpochs(Int epochs);
    
    // 设置学习率衰减因子
    void SetLRDecayFactor(Float factor);
    
    // 设置最小学习率
    void SetMinLearningRate(Float min_lr);
    
    // 单个样本的评估
    Float Evaluate(const Sample& sample, std::shared_ptr<Model> model, std::shared_ptr<Metric> metric) {
        Float pred = model->Forward(sample.features);
        metric->Add(pred, sample.label);
        return metric->Get();
    }
    
    // 数据集的评估
    Float Evaluate(const std::vector<Sample>& data, std::shared_ptr<Model> model, std::shared_ptr<Metric> metric) {
        metric->Reset();
        for (const auto& sample : data) {
            Float pred = model->Forward(sample.features);
            metric->Add(pred, sample.label);
        }
        return metric->Get();
    }
    
    // 打印训练超参数
    void PrintHyperparameters() const {
        std::cout << "======= 训练超参数 =======" << std::endl;
        std::cout << "批次大小 (batch_size): " << batch_size_ << std::endl;
        std::cout << "训练轮数 (epochs): " << epoch_num_ << std::endl;
        std::cout << "线程数 (num_threads): " << config_.num_threads << std::endl;
        std::cout << "学习率 (learning_rate): " << optimizer_->GetLearningRate() << std::endl;
        std::cout << "L2正则化 (l2_reg): " << optimizer_->GetL2Reg() << std::endl;
        std::cout << "学习率衰减轮数 (lr_decay_epochs): " << lr_decay_epochs_ << std::endl;
        std::cout << "学习率衰减因子 (lr_decay_factor): " << lr_decay_factor_ << std::endl;
        std::cout << "最小学习率 (min_learning_rate): " << min_learning_rate_ << std::endl;
        std::cout << "随机种子 (random_seed): " << random_seed_ << std::endl;
        std::cout << "打乱数据 (shuffle): " << (shuffle_ ? "true" : "false") << std::endl;
        std::cout << "日志打印频率 (log_interval): " << log_interval_ << std::endl;
        std::cout << "详细程度 (verbose): " << verbose_ << std::endl;
        std::cout << "模型保存路径 (model_save_path): " << config_.model_save_path << std::endl;
        std::cout << "======= 超参数结束 =======" << std::endl;
    }
    
private:
    std::shared_ptr<Model> model_;
    std::shared_ptr<Loss> loss_;
    std::shared_ptr<Optimizer> optimizer_;
    TrainerConfig config_;
    std::vector<std::shared_ptr<Metric>> metrics_;
    std::function<void(Int, Float, const std::vector<Float>&)> callback_;
    
    // 训练参数
    Int random_seed_;
    bool shuffle_;
    Int batch_size_;
    Int epoch_num_;
    Int verbose_;
    Int lr_decay_epochs_;
    Float lr_decay_factor_;
    Float min_learning_rate_;
    Int log_interval_;      // 日志打印频率
    
    // 线程池相关
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_flag_;
    
    // 初始化线程池
    void InitThreadPool();
    
    // 停止线程池
    void StopThreadPool();
    
    // 工作线程函数
    void WorkerThread();
    
    // 提交任务到线程池
    template<class F, class... Args>
    void Enqueue(F&& f, Args&&... args);
    
    // 训练一个批次
    Float TrainBatch(const Batch& batch);
    
    // 一个epoch的训练过程
    Float TrainEpoch(std::shared_ptr<DataReader> train_reader, Int current_epoch = 0);
    
    // 检查并处理不稳定的训练过程
    bool HandleUnstableTraining(Float current_loss, Float prev_loss);
};

} // namespace simpleflow

#endif // SIMPLEFLOW_TRAINER_H 