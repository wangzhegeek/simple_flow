#include "data_reader.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>

namespace simpleflow {

// 数据读取器基类
DataReader::DataReader(const String& file_path, Int batch_size, Int feature_dim)
    : file_path_(file_path), batch_size_(batch_size), feature_dim_(feature_dim) {
}

// LibSVM格式数据读取器
LibSVMReader::LibSVMReader(const String& file_path, Int batch_size, Int feature_dim)
    : DataReader(file_path, batch_size, feature_dim) {
    Reset();
}

bool LibSVMReader::NextBatch(Batch& batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    batch.clear();
    batch.reserve(batch_size_);
    
    String line;
    Int count = 0;
    
    while (count < batch_size_ && std::getline(file_stream_, line)) {
        std::istringstream iss(line);
        
        // 读取标签
        Float label;
        if (!(iss >> label)) {
            continue;  // 跳过格式不正确的行
        }
        
        // 读取特征
        Sample sample;
        sample.label = label;
        
        String feature_item;
        while (iss >> feature_item) {
            size_t colon_pos = feature_item.find(':');
            if (colon_pos != String::npos) {
                Int index = std::stoi(feature_item.substr(0, colon_pos));
                Float value = std::stof(feature_item.substr(colon_pos + 1));
                sample.features.emplace_back(index, value);
            }
        }
        
        // 按索引排序特征
        std::sort(sample.features.begin(), sample.features.end(), 
                 [](const SparseFeature& a, const SparseFeature& b) {
                     return a.index < b.index;
                 });
        
        batch.push_back(std::move(sample));
        ++count;
    }
    
    return !batch.empty();
}

void LibSVMReader::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    
    file_stream_.open(file_path_);
    if (!file_stream_.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path_);
    }
}

// Criteo格式数据读取器
CriteoReader::CriteoReader(const String& file_path, Int batch_size, Int feature_dim)
    : DataReader(file_path, batch_size, feature_dim) {
    Reset();
}

bool CriteoReader::NextBatch(Batch& batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    batch.clear();
    batch.reserve(batch_size_);
    
    String line;
    Int count = 0;
    
    while (count < batch_size_ && std::getline(file_stream_, line)) {
        std::istringstream iss(line);
        String token;
        
        // 读取标签（第一个字段）
        if (!std::getline(iss, token, '\t')) {
            continue;  // 跳过格式不正确的行
        }
        
        Sample sample;
        try {
            sample.label = std::stof(token);
            // 标签转换成-1和1，而不是0和1
            // if (sample.label == 0) {
            //     sample.label = -1;
            // }
        } catch (...) {
            continue;  // 跳过无效的标签
        }
        
        // 读取整数特征（接下来13个字段）
        Int field_id = 1;
        for (Int i = 0; i < 13; ++i) {
            if (!std::getline(iss, token, '\t')) {
                break;
            }
            
            if (!token.empty() && token != "") {
                try {
                    Int value = std::stoi(token);
                    if (value != 0) {  // 只添加非零特征
                        // 整数特征的索引从1到13
                        sample.features.emplace_back(field_id, value > 0 ? std::log(value + 1) : 0);
                    }
                } catch (...) {
                    // 对于无效值，不添加特征
                }
            }
            ++field_id;
        }
        
        // 读取类别特征（剩余26个字段）
        for (Int i = 0; i < 26; ++i) {
            if (!std::getline(iss, token, '\t')) {
                break;
            }
            
            if (!token.empty() && token != "") {
                // 使用一个更合适的哈希方法，保证哈希结果在合理范围内
                // 类别特征的基础索引从14到39
                Int base_index = field_id * 1000000;  // 为每个字段预留足够的哈希空间
                
                // 对类别特征进行哈希，确保不超过特征维度
                std::hash<String> hasher;
                Int hash_value = hasher(token) % 1000000;  // 限制在100万以内
                Int feature_index = base_index + hash_value;
                
                // 防止超出特征维度
                feature_index = feature_index % feature_dim_;
                if (feature_index < 14) {  // 避免与整数特征冲突
                    feature_index += 14;
                }
                
                sample.features.emplace_back(feature_index, 1.0);  // 类别特征值为1
            }
            ++field_id;
        }
        
        // 按索引排序特征
        std::sort(sample.features.begin(), sample.features.end(), 
                 [](const SparseFeature& a, const SparseFeature& b) {
                     return a.index < b.index;
                 });
        
        // 只有当样本包含特征时才添加到批次中
        if (!sample.features.empty()) {
            batch.push_back(std::move(sample));
            ++count;
        }
    }
    
    return !batch.empty();
}

void CriteoReader::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    
    file_stream_.open(file_path_);
    if (!file_stream_.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path_);
    }
}

// 工厂方法创建数据读取器
std::shared_ptr<DataReader> DataReader::Create(
    DataFormat format, 
    const String& file_path, 
    Int batch_size, 
    Int feature_dim) {
    
    switch (format) {
        case DataFormat::LIBSVM:
            return std::make_shared<LibSVMReader>(file_path, batch_size, feature_dim);
        case DataFormat::CRITEO:
            return std::make_shared<CriteoReader>(file_path, batch_size, feature_dim);
        default:
            throw std::runtime_error("Unsupported data format");
    }
}

// 数据格式转换辅助函数
DataFormat ParseDataFormat(const String& format_str) {
    if (format_str == "libsvm") {
        return DataFormat::LIBSVM;
    } else if (format_str == "criteo") {
        return DataFormat::CRITEO;
    } else if (format_str == "csv") {
        return DataFormat::CSV;
    }
    return DataFormat::Unknown;
}

// 添加一个工具方法用于标准化标签
void DataReader::StandardizeLabel(Sample& sample) {
    // 统一将-1标签转换为0，保证标签格式一致为0和1
    if (sample.label == -1) {
        sample.label = 0;
    }
}

// 在LibSVMReader::ReadSample中调用标签标准化方法
bool LibSVMReader::ReadSample(Sample& sample) {
    String line;
    if (!std::getline(file_stream_, line) || line.empty()) {
        return false;
    }
    
    std::istringstream iss(line);
    String token;
    
    // 读取标签
    if (!(iss >> token)) {
        return false;
    }
    
    try {
        sample.label = std::stof(token);
    } catch (const std::exception& e) {
        return false;
    }
    
    // 标准化标签
    StandardizeLabel(sample);
    
    // 读取特征
    sample.features.clear();
    while (iss >> token) {
        auto pos = token.find(':');
        if (pos == String::npos) {
            continue;
        }
        
        try {
            Int index = std::stoi(token.substr(0, pos));
            Float value = std::stof(token.substr(pos + 1));
            sample.features.push_back({index, value});
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    return true;
}

// 在CriteoReader::ReadSample中调用标签标准化方法
bool CriteoReader::ReadSample(Sample& sample) {
    String line;
    if (!std::getline(file_stream_, line) || line.empty()) {
        return false;
    }
    
    std::istringstream iss(line);
    String token;
    
    // 读取标签
    if (!(iss >> token)) {
        return false;
    }
    
    try {
        sample.label = std::stof(token);
    } catch (const std::exception& e) {
        return false;
    }
    
    // 标准化标签
    StandardizeLabel(sample);
    
    // 读取特征
    sample.features.clear();
    Int index = 0;
    while (iss >> token) {
        try {
            Float value = std::stof(token);
            sample.features.push_back({index, value});
            index++;
        } catch (const std::exception& e) {
            continue;
        }
    }
    
    return true;
}

} // namespace simpleflow 