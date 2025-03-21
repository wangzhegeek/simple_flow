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

DataReader::~DataReader() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

// LibSVM格式数据读取器
LibSVMReader::LibSVMReader(const String& file_path, Int batch_size, Int feature_dim)
    : DataReader(file_path, batch_size, feature_dim) {
    Reset();
}

LibSVMReader::~LibSVMReader() {
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

CriteoReader::~CriteoReader() {
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
        } catch (...) {
            continue;  // 跳过无效的标签
        }
        
        // 读取整数特征（接下来13个字段）
        Int field_id = 1;
        for (Int i = 0; i < 13; ++i) {
            if (!std::getline(iss, token, '\t')) {
                break;
            }
            
            if (!token.empty()) {
                try {
                    Int value = std::stoi(token);
                    // 将整数特征映射到索引
                    sample.features.emplace_back(field_id, value);
                } catch (...) {
                    // 对于无效值，使用默认值0
                    sample.features.emplace_back(field_id, 0);
                }
            }
            ++field_id;
        }
        
        // 读取类别特征（剩余26个字段）
        for (Int i = 0; i < 26; ++i) {
            if (!std::getline(iss, token, '\t')) {
                break;
            }
            
            if (!token.empty()) {
                // 对类别特征进行哈希
                std::hash<String> hasher;
                Int hash_value = hasher(token) % feature_dim_;
                // 类别特征索引从整数特征之后开始
                sample.features.emplace_back(field_id + hash_value, 1.0);
            }
            ++field_id;
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

} // namespace simpleflow 