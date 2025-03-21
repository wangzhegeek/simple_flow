#include "types.h"
#include "models/fm.h"
#include "data_reader.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"
#include "metric.h"
#include "utils/config_parser.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <map>
#include <algorithm>

using namespace simpleflow;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <配置文件路径>" << std::endl;
        return 1;
    }
    
    // 读取配置文件
    std::string config_path = argv[1];
    utils::ConfigParser config_parser;
    if (!config_parser.ParseFile(config_path)) {
        std::cerr << "无法解析配置文件: " << config_path << std::endl;
        return 1;
    }
    
    // 打印所有配置参数
    std::cout << "====== 从配置文件(" << config_path << ")读取的参数 ======" << std::endl;
    const auto& all_params = config_parser.GetAll();
    for (const auto& param : all_params) {
        std::cout << param.first << " = " << param.second << std::endl;
    }
    std::cout << "====== 配置参数结束 ======" << std::endl;
    
    // 数据路径
    String train_data_path = config_parser.GetString("train_data_path");
    String test_data_path = config_parser.GetString("test_data_path");
    String data_format_str = config_parser.GetString("data_format", "libsvm");
    DataFormat data_format = ParseDataFormat(data_format_str);
    
    // 模型参数
    Int feature_dim = config_parser.GetInt("feature_dim");
    Int embedding_size = config_parser.GetInt("embedding_size", 8);
    
    // 训练参数
    Int batch_size = config_parser.GetInt("batch_size", 128);
    Int epochs = config_parser.GetInt("epochs", 10);
    Float learning_rate = config_parser.GetFloat("learning_rate", 0.01);
    Float l2_reg = config_parser.GetFloat("l2_reg", 0.0001);
    Int log_interval = config_parser.GetInt("log_interval", 100);
    Int num_threads = config_parser.GetInt("num_threads", 1);
    
    std::cout << "Debug - 读取到的log_interval参数值: " << log_interval << std::endl;
    
    // 激活函数
    String activation_type_str = config_parser.GetString("activation_type", "sigmoid");
    ActivationType activation_type = ParseActivationType(activation_type_str);
    
    // 损失函数
    String loss_type_str = config_parser.GetString("loss_type", "logloss");
    LossType loss_type = ParseLossType(loss_type_str);
    
    // 优化器
    String optimizer_type_str = config_parser.GetString("optimizer_type", "sgd");
    OptimizerType optimizer_type = ParseOptimizerType(optimizer_type_str);
    
    // 创建数据读取器
    auto train_reader = DataReader::Create(data_format, train_data_path, batch_size, feature_dim);
    auto test_reader = DataReader::Create(data_format, test_data_path, batch_size, feature_dim);
    
    // 创建FM模型
    auto activation = Activation::Create(activation_type);
    auto loss = Loss::Create(loss_type);
    
    auto model = std::make_shared<FMModel>(feature_dim, embedding_size, activation, loss);
    
    // 创建优化器
    auto optimizer = Optimizer::Create(optimizer_type, learning_rate, l2_reg);
    
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
    trainer_config.num_threads = num_threads;
    trainer_config.verbose = config_parser.GetBool("verbose", true);
    trainer_config.model_save_path = config_parser.GetString("model_save_path", "./model/fm_model.bin");
    trainer_config.lr_decay_epochs = config_parser.GetInt("lr_decay_epochs", 5);
    trainer_config.lr_decay_factor = config_parser.GetFloat("lr_decay_factor", 0.5f);
    trainer_config.min_learning_rate = config_parser.GetFloat("min_learning_rate", 1e-6f);
    trainer_config.log_interval = log_interval;

    std::cout << "Debug - 设置到TrainerConfig中的log_interval值: " << trainer_config.log_interval << std::endl;
    
    // 创建训练器
    Trainer trainer(model, loss, optimizer, trainer_config);
    
    // 添加评估指标
    for (const auto& metric : metrics) {
        trainer.AddMetric(metric);
    }
    
    // 设置训练回调
    trainer.SetCallback([](Int epoch, Float train_loss, const std::vector<Float>& metric_values) {
        std::cout << "Epoch " << epoch << ", Train Loss: " << train_loss;
        
        if (!metric_values.empty()) {
            std::cout << ", Accuracy: " << metric_values[0];
        }
        if (metric_values.size() > 1) {
            std::cout << ", AUC: " << metric_values[1];
        }
        if (metric_values.size() > 2) {
            std::cout << ", LogLoss: " << metric_values[2];
        }
        
        std::cout << std::endl;
    });
    
    // 记录训练开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 训练模型
    trainer.Train(train_reader, test_reader);
    
    // 计算训练时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "训练完成，用时 " << duration << " 秒" << std::endl;
    
    // 评估最终模型
    std::cout << "最终模型评估：" << std::endl;
    trainer.Evaluate(test_reader, metrics);
    
    for (const auto& metric : metrics) {
        std::cout << metric->GetName() << ": " << metric->Get() << std::endl;
    }
    
    return 0;
} 