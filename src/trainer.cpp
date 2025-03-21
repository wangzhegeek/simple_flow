#include "trainer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cmath>

namespace simpleflow {

Trainer::Trainer(std::shared_ptr<Model> model, 
                 std::shared_ptr<Loss> loss,
                 std::shared_ptr<Optimizer> optimizer,
                 const TrainerConfig& config)
    : model_(model), loss_(loss), optimizer_(optimizer), config_(config), stop_flag_(false),
      random_seed_(config.random_seed), shuffle_(config.shuffle),
      batch_size_(config.batch_size), epoch_num_(config.epochs),
      verbose_(config.verbose), lr_decay_epochs_(config.lr_decay_epochs),
      lr_decay_factor_(config.lr_decay_factor), min_learning_rate_(config.min_learning_rate),
      log_interval_(config.log_interval) {
    if (!model_) {
        throw std::invalid_argument("Model cannot be null");
    }
    if (!loss_) {
        throw std::invalid_argument("Loss cannot be null");
    }
    if (!optimizer_) {
        throw std::invalid_argument("Optimizer cannot be null");
    }
    
    // 初始化线程池
    InitThreadPool();
}

Trainer::~Trainer() {
    StopThreadPool();
}

void Trainer::Train(std::shared_ptr<DataReader> train_reader, std::shared_ptr<DataReader> valid_reader) {
    if (!train_reader) {
        throw std::invalid_argument("Training data reader cannot be null");
    }
    
    // 打印所有训练超参数
    if (verbose_ > 0) {
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
    
    // 原始学习率
    Float initial_learning_rate = optimizer_->GetLearningRate();
    
    // 训练开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证指标
    Float best_validation_metric = 0.0f;
    int no_improvement_count = 0;
    
    // 上一轮的训练损失，用于检测不稳定训练
    Float prev_epoch_loss = std::numeric_limits<Float>::max();
    
    // 循环训练epoch_num_轮
    for (Int epoch = 0; epoch < epoch_num_; ++epoch) {
        // 每隔lr_decay_epochs_轮次，降低学习率
        if (epoch > 0 && epoch % lr_decay_epochs_ == 0) {
            Float current_lr = optimizer_->GetLearningRate();
            Float new_lr = current_lr * lr_decay_factor_;
            
            // 确保学习率不低于最小值
            if (new_lr >= min_learning_rate_) {
                optimizer_->SetLearningRate(new_lr);
                if (verbose_ > 0) {
                    std::cout << "学习率降低至: " << new_lr << std::endl;
                }
            }
        }
        
        // 训练一个epoch
        Float train_loss = TrainEpoch(train_reader, epoch);
        
        // 如果当前损失突然增大，可能是训练不稳定的信号
        if (train_loss > prev_epoch_loss * 1.5f && prev_epoch_loss < 10.0f) { // 只有当前一次损失够小时才考虑不稳定
            std::cerr << "警告: 损失值突然增大，可能存在训练不稳定问题" << std::endl;
            
            // 降低学习率
            Float current_lr = optimizer_->GetLearningRate();
            Float new_lr = current_lr * 0.8f; // 更温和的降低
            
            if (new_lr >= min_learning_rate_) {
                optimizer_->SetLearningRate(new_lr);
                std::cout << "自动降低学习率至: " << new_lr << std::endl;
            }
        }
        
        // 更新上一轮损失
        prev_epoch_loss = train_loss;
        
        // 评估模型
        std::vector<Float> metric_values;
        if (valid_reader) {
            Evaluate(valid_reader, metrics_);
            for (const auto& metric : metrics_) {
                metric_values.push_back(metric->Get());
            }
            
            // 记录最佳验证指标
            if (metrics_.size() > 1) {
                Float current_validation_metric = metrics_[1]->Get(); // 假设第二个指标是我们关注的
                if (current_validation_metric > best_validation_metric) {
                    best_validation_metric = current_validation_metric;
                    no_improvement_count = 0;
                } else {
                    ++no_improvement_count;
                    
                    // 如果连续多轮没有改善，考虑降低学习率，更温和的策略
                    if (no_improvement_count >= 3) { // 需要更多轮次来确认没有提升
                        Float current_lr = optimizer_->GetLearningRate();
                        Float new_lr = current_lr * 0.8f; // 更温和的降低
                        
                        if (new_lr >= min_learning_rate_) {
                            optimizer_->SetLearningRate(new_lr);
                            std::cout << "验证指标未提升，降低学习率至: " << new_lr << std::endl;
                            no_improvement_count = 0;
                        }
                    }
                }
            }
        }
        
        // 回调函数
        if (callback_) {
            callback_(epoch + 1, train_loss, metric_values);
        }
        
        // 保存模型
        if (!config_.model_save_path.empty()) {
            String model_path = config_.model_save_path;
            if (config_.model_save_path.find(".bin") == String::npos) {
                model_path += "_epoch" + std::to_string(epoch + 1) + ".bin";
            }
            model_->Save(model_path);
        }
    }
    
    // 计算总训练时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    if (verbose_ > 0) {
        std::cout << "训练完成，总用时: " << duration.count() << " 秒" << std::endl;
    }
    
    // 恢复原始学习率
    optimizer_->SetLearningRate(initial_learning_rate);
}

Float Trainer::TrainEpoch(std::shared_ptr<DataReader> train_reader, Int current_epoch) {
    train_reader->Reset();
    
    Float total_loss = 0.0;
    Int batch_count = 0;
    
    // 重置指标
    for (auto& metric : metrics_) {
        metric->Reset();
    }
    
    Batch batch;
    while (train_reader->NextBatch(batch)) {
        Float batch_loss = TrainBatch(batch);
        total_loss += batch_loss;
        ++batch_count;
        
        // 前向传播获取预测结果，用于更新指标
        FloatVector predictions;
        model_->Forward(batch, predictions);
        
        // 计算标签，将-1转换为0，适应sigmoid激活函数
        FloatVector targets(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            // 转换标签，-1 -> 0, 1 -> 1
            targets[i] = (batch[i].label == -1) ? 0.0f : batch[i].label;
        }
        
        // 更新指标
        for (auto& metric : metrics_) {
            metric->Add(predictions, targets);
        }
        
        if (verbose_ > 0 && batch_count % log_interval_ == 0) {
            std::cout << "Epoch " << (current_epoch + 1) << "/" << epoch_num_
                      << ", Batch " << batch_count << ", Loss: " << std::fixed << std::setprecision(5) << batch_loss;
                      
            std::cout << std::endl;
        }
    }
    
    Float avg_loss = (batch_count > 0) ? total_loss / batch_count : 0.0f;
    
    // 只有在没有设置回调函数时才在这里打印训练结果，避免重复输出
    if (verbose_ > 0 && !callback_) {
        std::cout << "Epoch " << (current_epoch + 1) << "/" << epoch_num_
                  << ", Loss: " << std::fixed << std::setprecision(5) << avg_loss;
        
        // 打印指标
        for (auto& metric : metrics_) {
            std::cout << ", " << metric->GetName() << ": "
                      << std::fixed << std::setprecision(5) << metric->Get();
        }
        std::cout << std::endl;
    }
    
    return avg_loss;
}

Float Trainer::TrainBatch(const Batch& batch) {
    if (batch.empty()) {
        return 0.0;
    }
    
    // 前向传播
    FloatVector predictions;
    model_->Forward(batch, predictions);
    
    // 计算标签，将-1转换为0，适应sigmoid激活函数
    FloatVector targets(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        // 转换标签，-1 -> 0, 1 -> 1
        targets[i] = (batch[i].label == -1) ? 0.0f : batch[i].label;
    }
    
    // 检查预测值是否在合理范围内（对于二分类问题）
    // 对于LogLoss，预测值应在[0,1]范围内
    if (config_.verbose && batch.size() > 0) {
        Int out_of_range = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] < 0.0 || predictions[i] > 1.0) {
                out_of_range++;
            }
        }
        if (out_of_range > 0 && batch.size() % 10 == 0) {
            std::cout << "Warning: " << out_of_range << " predictions out of range [0,1]" << std::endl;
        }
    }
    
    // 预测值应该在合理范围内，否则可能出现数值不稳定问题
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (std::isnan(predictions[i]) || std::isinf(predictions[i])) {
            std::cerr << "警告: 预测值包含NaN或Inf值，可能出现数值不稳定问题" << std::endl;
            predictions[i] = 0.5f; // 将异常值替换为中间值
        }
    }
    
    // 计算损失
    Float batch_loss = loss_->Compute(predictions, targets);
    
    // 检查损失值是否有效
    if (std::isnan(batch_loss) || std::isinf(batch_loss)) {
        std::cerr << "警告: 损失值为NaN或Inf，跳过此批次更新" << std::endl;
        return 0.0f;
    }
    
    // 确保损失值为非负数
    if (batch_loss < 0) {
        batch_loss = std::abs(batch_loss);
    }
    
    // 计算梯度
    FloatVector gradients;
    loss_->Gradient(predictions, targets, gradients);
    
    // 反向传播
    model_->Backward(batch, gradients, optimizer_);
    
    return batch_loss;
}

void Trainer::Evaluate(std::shared_ptr<DataReader> data_reader, const std::vector<std::shared_ptr<Metric>>& metrics) {
    if (!data_reader) {
        throw std::invalid_argument("Data reader cannot be null");
    }
    
    data_reader->Reset();
    
    // 重置所有指标
    for (auto& metric : metrics) {
        metric->Reset();
    }
    
    Batch batch;
    while (data_reader->NextBatch(batch)) {
        // 前向传播
        FloatVector predictions;
        model_->Forward(batch, predictions);
        
        // 计算标签，将-1转换为0，适应sigmoid激活函数
        FloatVector targets(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            // 转换标签，-1 -> 0, 1 -> 1
            targets[i] = (batch[i].label == -1) ? 0.0f : batch[i].label;
        }
        
        // 更新所有指标
        for (auto& metric : metrics) {
            metric->Add(predictions, targets);
        }
    }
}

void Trainer::Predict(std::shared_ptr<DataReader> data_reader, FloatVector& predictions) {
    if (!data_reader) {
        throw std::invalid_argument("Data reader cannot be null");
    }
    
    data_reader->Reset();
    
    predictions.clear();
    
    Batch batch;
    while (data_reader->NextBatch(batch)) {
        // 前向传播
        FloatVector batch_predictions;
        model_->Forward(batch, batch_predictions);
        
        // 添加到总预测结果
        predictions.insert(predictions.end(), batch_predictions.begin(), batch_predictions.end());
    }
}

void Trainer::AddMetric(std::shared_ptr<Metric> metric) {
    if (metric) {
        metrics_.push_back(metric);
    }
}

void Trainer::SetCallback(std::function<void(Int, Float, const std::vector<Float>&)> callback) {
    callback_ = callback;
}

void Trainer::InitThreadPool() {
    Int num_threads = config_.num_threads > 0 ? config_.num_threads : 1;
    
    for (Int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back(&Trainer::WorkerThread, this);
    }
}

void Trainer::StopThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_flag_ = true;
    }
    
    condition_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void Trainer::WorkerThread() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { 
                return stop_flag_ || !task_queue_.empty(); 
            });
            
            if (stop_flag_ && task_queue_.empty()) {
                return;
            }
            
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }
        
        task();
    }
}

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

bool Trainer::HandleUnstableTraining(Float current_loss, Float prev_loss) {
    if (std::isnan(current_loss) || std::isinf(current_loss)) {
        std::cerr << "警告: 损失值为NaN或Inf，尝试降低学习率重新训练" << std::endl;
        
        // 将学习率降低为原来的一半
        Float current_lr = optimizer_->GetLearningRate();
        Float new_lr = current_lr * 0.5f;
        
        // 确保学习率不低于最小值
        if (new_lr >= min_learning_rate_) {
            optimizer_->SetLearningRate(new_lr);
            return true; // 表示需要重新训练
        } else {
            std::cerr << "学习率已达最小值，无法继续降低" << std::endl;
            return false;
        }
    }
    
    // 如果当前损失突然增大，可能是训练不稳定的信号
    if (current_loss > prev_loss * 1.5f) {
        std::cerr << "警告: 损失值突然增大，可能存在训练不稳定问题" << std::endl;
        
        // 降低学习率
        Float current_lr = optimizer_->GetLearningRate();
        Float new_lr = current_lr * 0.5f;
        
        if (new_lr >= min_learning_rate_) {
            optimizer_->SetLearningRate(new_lr);
            std::cout << "自动降低学习率至: " << new_lr << std::endl;
        }
    }
    
    return false; // 不需要重新训练
}

void Trainer::SetRandomSeed(Int seed) {
    random_seed_ = seed;
}

void Trainer::SetShuffle(bool shuffle) {
    shuffle_ = shuffle;
}

void Trainer::SetBatchSize(Int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    batch_size_ = batch_size;
}

void Trainer::SetEpochNum(Int epoch_num) {
    if (epoch_num <= 0) {
        throw std::invalid_argument("Epoch number must be positive");
    }
    epoch_num_ = epoch_num;
}

void Trainer::SetVerbose(Int verbose) {
    verbose_ = verbose;
}

void Trainer::SetLRDecayEpochs(Int epochs) {
    if (epochs <= 0) {
        throw std::invalid_argument("LR decay epochs must be positive");
    }
    lr_decay_epochs_ = epochs;
}

void Trainer::SetLRDecayFactor(Float factor) {
    if (factor <= 0.0f || factor >= 1.0f) {
        throw std::invalid_argument("LR decay factor must be between 0 and 1");
    }
    lr_decay_factor_ = factor;
}

void Trainer::SetMinLearningRate(Float min_lr) {
    if (min_lr <= 0.0f) {
        throw std::invalid_argument("Minimum learning rate must be positive");
    }
    min_learning_rate_ = min_lr;
}

} // namespace simpleflow 