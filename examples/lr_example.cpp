#include "simpleflow/types.h"
#include "simpleflow/models/lr.h"
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
    // 解析配置文件路径
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    
    std::string config_file = argv[1];
    utils::ConfigParser config_parser;
    if (!config_parser.ParseFile(config_file)) {
        std::cerr << "Failed to parse config file: " << config_file << std::endl;
        return 1;
    }
    
    // 调试输出：打印所有配置参数
    std::cout << "====== 从配置文件(" << config_file << ")读取的参数 ======" << std::endl;
    const auto& all_params = config_parser.GetAll();
    for (const auto& param : all_params) {
        std::cout << param.first << " = " << param.second << std::endl;
    }
    std::cout << "====== 配置参数结束 ======" << std::endl;
    
    // 读取配置
    String data_format = config_parser.GetString("data_format", "libsvm");
    String train_data_path = config_parser.GetString("train_data_path");
    String test_data_path = config_parser.GetString("test_data_path");
    Int feature_dim = config_parser.GetInt("feature_dim", 0);
    Int batch_size = config_parser.GetInt("batch_size", 128);
    Int epochs = config_parser.GetInt("epochs", 10);
    Float learning_rate = config_parser.GetFloat("learning_rate", 0.01);
    Float l2_reg = config_parser.GetFloat("l2_reg", 0.001);
    Int num_threads = config_parser.GetInt("num_threads", 1);
    bool verbose = config_parser.GetBool("verbose", true);
    String model_save_path = config_parser.GetString("model_save_path", "./model/lr_model.bin");
    
    // 读取学习率衰减相关参数
    Int lr_decay_epochs = config_parser.GetInt("lr_decay_epochs", 3);
    Float lr_decay_factor = config_parser.GetFloat("lr_decay_factor", 0.5);
    Float min_learning_rate = config_parser.GetFloat("min_learning_rate", 1e-6);
    
    // 读取日志打印频率
    Int log_interval = config_parser.GetInt("log_interval", 100);
    std::cout << "Debug - 读取到的log_interval参数值: " << log_interval << std::endl;
    
    // 读取激活函数、损失函数和优化器类型
    String activation_type_str = config_parser.GetString("activation_type", "sigmoid");
    String loss_type_str = config_parser.GetString("loss_type", "logloss");
    String optimizer_type_str = config_parser.GetString("optimizer_type", "sgd");
    
    // 检查必要的参数
    if (train_data_path.empty() || feature_dim == 0) {
        std::cerr << "Missing required parameters: train_data_path or feature_dim" << std::endl;
        return 1;
    }
    
    // 创建数据读取器
    DataFormat format = ParseDataFormat(data_format);
    auto train_reader = DataReader::Create(format, train_data_path, batch_size, feature_dim);
    std::shared_ptr<DataReader> test_reader = nullptr;
    if (!test_data_path.empty()) {
        test_reader = DataReader::Create(format, test_data_path, batch_size, feature_dim);
    }
    
    // 创建LR模型
    ActivationType activation_type = ParseActivationType(activation_type_str);
    auto activation = Activation::Create(activation_type);
    auto model = std::make_shared<LRModel>(feature_dim, activation);
    model->Init();
    
    // 创建损失函数、优化器和训练器
    LossType loss_type = ParseLossType(loss_type_str);
    auto loss = Loss::Create(loss_type);
    
    OptimizerType optimizer_type = ParseOptimizerType(optimizer_type_str);
    auto optimizer = Optimizer::Create(optimizer_type, learning_rate, l2_reg);
    
    TrainerConfig trainer_config;
    trainer_config.batch_size = batch_size;
    trainer_config.epochs = epochs;
    trainer_config.num_threads = num_threads;
    trainer_config.verbose = verbose;
    trainer_config.model_save_path = model_save_path;
    trainer_config.lr_decay_epochs = lr_decay_epochs;      // 设置学习率衰减轮数
    trainer_config.lr_decay_factor = lr_decay_factor;      // 设置学习率衰减因子
    trainer_config.min_learning_rate = min_learning_rate;  // 设置最小学习率
    trainer_config.log_interval = log_interval;           // 设置日志打印频率
    std::cout << "Debug - 设置到TrainerConfig中的log_interval值: " << trainer_config.log_interval << std::endl;
    
    auto trainer = std::make_shared<Trainer>(model, loss, optimizer, trainer_config);
    
    // 添加评估指标
    trainer->AddMetric(std::make_shared<AccuracyMetric>());
    trainer->AddMetric(std::make_shared<AUCMetric>());
    trainer->AddMetric(std::make_shared<LogLossMetric>());
    
    // 设置训练回调
    trainer->SetCallback([log_interval](Int epoch, Float train_loss, const std::vector<Float>& metric_values) {
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
    
    // 训练模型
    auto start_time = std::chrono::high_resolution_clock::now();
    trainer->Train(train_reader, test_reader);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "Training completed in " << duration << " seconds" << std::endl;
    
    // 在测试集上进行完整评估
    if (test_reader) {
        std::cout << "\n========== 在测试集上的评估结果 ==========\n";
        
        // 创建评估指标
        std::vector<std::shared_ptr<Metric>> test_metrics;
        test_metrics.push_back(std::make_shared<AccuracyMetric>());
        test_metrics.push_back(std::make_shared<AUCMetric>());
        test_metrics.push_back(std::make_shared<LogLossMetric>());
        
        // 在测试集上评估模型
        trainer->Evaluate(test_reader, test_metrics);
        
        // 输出测试集上的指标结果
        for (const auto& metric : test_metrics) {
            std::cout << metric->GetName() << ": " << std::fixed << std::setprecision(5) << metric->Get() << std::endl;
        }
        std::cout << "==========================================\n";
    }
    
    // 保存模型
    model->Save(model_save_path);
    
    return 0;
} 