#include "simpleflow/types.h"
#include "simpleflow/models/fm.h"
#include "simpleflow/data_reader.h"
#include "simpleflow/activation.h"
#include "simpleflow/loss.h"
#include "simpleflow/optimizer.h"
#include "simpleflow/trainer.h"
#include "simpleflow/metric.h"
#include "simpleflow/utils/config_parser.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

using namespace simpleflow;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <配置文件路径>" << std::endl;
        return 1;
    }
    
    // 读取配置文件
    std::string config_path = argv[1];
    ConfigParser config_parser(config_path);
    
    // 打印所有配置参数
    std::cout << "====== 从配置文件(" << config_path << ")读取的参数 ======" << std::endl;
    config_parser.PrintAllParameters();
    std::cout << "====== 配置参数结束 ======" << std::endl;
    
    // 数据路径
    std::string train_data_path = config_parser.GetString("train_data_path");
    std::string test_data_path = config_parser.GetString("test_data_path");
    std::string data_format = config_parser.GetString("data_format", "libsvm");
    
    // 模型参数
    Int feature_dim = config_parser.GetInt("feature_dim");
    Int embedding_size = config_parser.GetInt("embedding_size", 8);
    
    // 训练参数
    Int batch_size = config_parser.GetInt("batch_size", 128);
    Int epochs = config_parser.GetInt("epochs", 10);
    Float learning_rate = config_parser.GetFloat("learning_rate", 0.01);
    Float l2_reg = config_parser.GetFloat("l2_reg", 0.0001);
    Int log_interval = config_parser.GetInt("log_interval", 100);
    
    std::cout << "Debug - 读取到的log_interval参数值: " << log_interval << std::endl;
    
    // 激活函数
    std::string activation_type_str = config_parser.GetString("activation_type", "sigmoid");
    ActivationType activation_type = ParseActivationType(activation_type_str);
    
    // 损失函数
    std::string loss_type_str = config_parser.GetString("loss_type", "logloss");
    LossType loss_type = ParseLossType(loss_type_str);
    
    // 优化器
    std::string optimizer_type_str = config_parser.GetString("optimizer_type", "sgd");
    OptimizerType optimizer_type = ParseOptimizerType(optimizer_type_str);
    
    // 创建数据读取器
    std::shared_ptr<DataReader> train_reader = std::make_shared<DataReader>(train_data_path, data_format);
    std::shared_ptr<DataReader> test_reader = std::make_shared<DataReader>(test_data_path, data_format);
    
    // 创建FM模型
    std::shared_ptr<Activation> activation = Activation::Create(activation_type);
    std::shared_ptr<Loss> loss = Loss::Create(loss_type);
    
    std::shared_ptr<FMModel> model = std::make_shared<FMModel>(feature_dim, embedding_size, activation, loss);
    
    // 创建优化器
    std::shared_ptr<Optimizer> optimizer = Optimizer::Create(optimizer_type, learning_rate, l2_reg);
    
    // 创建评估指标
    std::vector<std::shared_ptr<Metric>> metrics = {
        std::make_shared<AccuracyMetric>(),
        std::make_shared<AUCMetric>(),
        std::make_shared<LogLossMetric>()
    };
    
    // 构建训练配置
    TrainerConfig trainer_config;
    trainer_config.batch_size = batch_size;
    trainer_config.epochs = epochs;
    trainer_config.learning_rate = learning_rate;
    trainer_config.l2_reg = l2_reg;
    trainer_config.verbose = config_parser.GetBool("verbose", true);
    trainer_config.model_save_path = config_parser.GetString("model_save_path", "./model/fm_model.bin");
    trainer_config.lr_decay_epochs = config_parser.GetInt("lr_decay_epochs", 5);
    trainer_config.lr_decay_factor = config_parser.GetFloat("lr_decay_factor", 0.5);
    trainer_config.min_learning_rate = config_parser.GetFloat("min_learning_rate", 1e-6);
    trainer_config.log_interval = log_interval;

    std::cout << "Debug - 设置到TrainerConfig中的log_interval值: " << trainer_config.log_interval << std::endl;
    
    // 创建训练器
    Trainer trainer(model, loss, optimizer, trainer_config);
    
    // 打印训练参数
    trainer.PrintHyperparameters();
    
    // 使用自定义训练逻辑
    auto train_data = train_reader->ReadAll();
    auto test_data = test_reader->ReadAll();
    
    // 记录训练开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (Int epoch = 0; epoch < epochs; ++epoch) {
        Float epoch_loss = 0.0;
        Int batch_count = 0;
        
        // 打乱训练数据
        if (trainer_config.shuffle) {
            train_reader->Shuffle(train_data);
        }
        
        // 按批次训练
        for (Int i = 0; i < train_data.size(); i += batch_size) {
            Int current_batch_size = std::min(batch_size, (Int)(train_data.size() - i));
            Float batch_loss = 0.0;
            
            for (Int j = 0; j < current_batch_size; ++j) {
                const auto& sample = train_data[i + j];
                
                // 前向传播
                Float pred = model->Forward(sample.features);
                
                // 计算损失
                Float sample_loss = loss->Compute(pred, sample.label);
                batch_loss += sample_loss;
                
                // 反向传播（使用FM的新接口，直接传递预测值和目标值）
                model->Backward(sample.features, pred, sample.label, optimizer.get());
            }
            
            batch_loss /= current_batch_size;
            epoch_loss += batch_loss;
            batch_count++;
            
            // 打印批次信息
            if (batch_count % trainer_config.log_interval == 0 && trainer_config.verbose) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                          << ", Batch " << batch_count
                          << ", Loss: " << batch_loss
                          << std::endl;
            }
        }
        
        epoch_loss /= batch_count;
        
        // 评估训练集
        std::map<std::string, Float> train_metrics;
        for (const auto& metric : metrics) {
            Float value = trainer.Evaluate(train_data, model, metric);
            train_metrics[metric->GetName()] = value;
        }
        
        // 打印训练信息
        std::cout << "Epoch " << (epoch + 1)
                  << ", Train Loss: " << epoch_loss;
        
        for (const auto& [name, value] : train_metrics) {
            std::cout << ", " << name << ": " << value;
        }
        std::cout << std::endl;
        
        // 学习率衰减
        if ((epoch + 1) % trainer_config.lr_decay_epochs == 0) {
            Float old_lr = optimizer->GetLearningRate();
            Float new_lr = std::max(old_lr * trainer_config.lr_decay_factor, trainer_config.min_learning_rate);
            optimizer->SetLearningRate(new_lr);
            std::cout << "学习率降低至: " << new_lr << std::endl;
        }
        
        // 评估验证集
        bool metrics_improved = false;
        std::map<std::string, Float> prev_test_metrics;
        std::map<std::string, Float> test_metrics;
        
        for (const auto& metric : metrics) {
            Float value = trainer.Evaluate(test_data, model, metric);
            test_metrics[metric->GetName()] = value;
        }
        
        // 根据验证集性能调整学习率
        if (epoch > 0 && test_metrics["Accuracy"] <= prev_test_metrics["Accuracy"]) {
            Float old_lr = optimizer->GetLearningRate();
            Float new_lr = std::max(old_lr * 0.8, trainer_config.min_learning_rate);
            optimizer->SetLearningRate(new_lr);
            std::cout << "验证指标未提升，降低学习率至: " << new_lr << std::endl;
        } else {
            metrics_improved = true;
        }
        
        prev_test_metrics = test_metrics;
        
        // 保存模型
        if (metrics_improved && !trainer_config.model_save_path.empty()) {
            model->Save(trainer_config.model_save_path);
        }
        
        // 如果学习率太小，提前停止
        if (optimizer->GetLearningRate() <= trainer_config.min_learning_rate) {
            std::cout << "学习率太小，提前停止训练" << std::endl;
            break;
        }
    }
    
    // 计算训练时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "训练完成，用时 " << duration << " 秒" << std::endl;
    
    // 评估最终模型
    std::cout << "最终模型评估：" << std::endl;
    for (const auto& metric : metrics) {
        Float train_value = trainer.Evaluate(train_data, model, metric);
        Float test_value = trainer.Evaluate(test_data, model, metric);
        
        std::cout << metric->GetName() << ": "
                  << "Train=" << train_value << ", "
                  << "Test=" << test_value << std::endl;
    }
    
    return 0;
} 