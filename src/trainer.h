#ifndef SIMPLE_FLOW_TRAINER_H_
#define SIMPLE_FLOW_TRAINER_H_

#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <limits>

#include "types.h"

// TrainerConfig定义
namespace simpleflow {
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
}

#include "model.h"
#include "loss.h"
#include "optimizer.h"
#include "data_reader.h"
#include "metric.h"

namespace simpleflow {

class Trainer {
public:
    Trainer(std::shared_ptr<Model> model, 
            std::shared_ptr<Loss> loss,
            std::shared_ptr<Optimizer> optimizer,
            const TrainerConfig& config);
    
    ~Trainer();
    
    // 训练模型
    void Train(std::shared_ptr<DataReader> train_reader, std::shared_ptr<DataReader> valid_reader = nullptr);
    
    // 评估模型
    void Evaluate(std::shared_ptr<DataReader> data_reader, const std::vector<std::shared_ptr<Metric>>& metrics);
    
    // 预测
    void Predict(std::shared_ptr<DataReader> data_reader, FloatVector& predictions);
    
    // 添加评估指标
    void AddMetric(std::shared_ptr<Metric> metric);
    
    // 设置回调函数
    void SetCallback(std::function<void(Int, Float, const std::vector<Float>&)> callback);
    
    // 处理不稳定训练
    bool HandleUnstableTraining(Float current_loss, Float prev_loss);
    
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
    
private:
    // 训练一个epoch
    Float TrainEpoch(std::shared_ptr<DataReader> train_reader, Int current_epoch);
    
    // 训练一个批次
    Float TrainBatch(const Batch& batch);
    
    // 内部训练批次实现（用于多线程）
    Float TrainBatchInternal(const Batch& batch, FloatVector* out_predictions = nullptr);
    
    // 初始化线程池
    void InitThreadPool();
    
    // 停止线程池
    void StopThreadPool();
    
    // 工作线程函数
    void WorkerThread();
    
    // 向线程池添加任务
    template<class F, class... Args>
    void Enqueue(F&& f, Args&&... args);
    
    std::shared_ptr<Model> model_;
    std::shared_ptr<Loss> loss_;
    std::shared_ptr<Optimizer> optimizer_;
    TrainerConfig config_;
    
    // 是否停止线程池
    bool stop_flag_;
    
    // 线程池相关
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    
    // 评估指标
    std::vector<std::shared_ptr<Metric>> metrics_;
    
    // 回调函数
    std::function<void(Int, Float, const std::vector<Float>&)> callback_;
    
    // 训练超参数
    Int random_seed_;
    bool shuffle_;
    Int batch_size_;
    Int epoch_num_;
    Int verbose_;
    Int lr_decay_epochs_;
    Float lr_decay_factor_;
    Float min_learning_rate_;
    Int log_interval_;
};

// 线程池任务入队实现
template<class F, class... Args>
void Trainer::Enqueue(F&& f, Args&&... args) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_flag_) {
            throw std::runtime_error("Enqueue on stopped ThreadPool");
        }
        
        task_queue_.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }
    
    condition_.notify_one();
}

} // namespace simpleflow

#endif // SIMPLE_FLOW_TRAINER_H_ 