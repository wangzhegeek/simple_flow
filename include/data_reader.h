#ifndef SIMPLEFLOW_DATA_READER_H
#define SIMPLEFLOW_DATA_READER_H

#include "types.h"
#include <fstream>
#include <mutex>
#include <memory>

namespace simpleflow {

// 数据格式类型
enum class DataFormat {
    LIBSVM,
    CRITEO,
    CSV,
    Unknown
};

// 数据读取器基类
class DataReader {
public:
    DataReader(const String& file_path, Int batch_size, Int feature_dim);
    virtual ~DataReader() = default;

    // 获取下一批次数据
    virtual bool NextBatch(Batch& batch) = 0;
    
    // 重置读取器到开始位置
    virtual void Reset() = 0;
    
    // 获取特征维度
    Int GetFeatureDim() const { return feature_dim_; }
    
    // 打开数据文件
    virtual bool Open(const String& file_path) {
        file_path_ = file_path;
        Reset();
        return file_stream_.is_open();
    }
    
    // 读取一个样本
    virtual bool ReadSample(Sample& sample) = 0;
    
    // 创建批次
    virtual Batch CreateBatch(Int batch_size) {
        Batch batch;
        batch.reserve(batch_size);
        Sample sample;
        for (Int i = 0; i < batch_size && ReadSample(sample); ++i) {
            batch.push_back(sample);
        }
        return batch;
    }
    
    // 工厂方法创建指定格式的数据读取器
    static std::shared_ptr<DataReader> Create(
        DataFormat format, 
        const String& file_path, 
        Int batch_size, 
        Int feature_dim);

protected:
    String file_path_;
    Int batch_size_;
    Int feature_dim_;
    std::ifstream file_stream_;

    // 用于标准化标签的工具方法
    void StandardizeLabel(Sample& sample);
};

// LibSVM格式数据读取器
class LibSVMReader : public DataReader {
public:
    LibSVMReader(const String& file_path, Int batch_size, Int feature_dim);
    ~LibSVMReader() override = default;
    
    // 获取下一批次数据
    bool NextBatch(Batch& batch) override;
    
    // 重置读取器到开始位置
    void Reset() override;
    
    // 读取一个样本
    bool ReadSample(Sample& sample) override;
    
    // 使用基类的Open方法
    bool Open(const String& file_path) override {
        return DataReader::Open(file_path);
    }
    
private:
    std::mutex mutex_;
};

// Criteo格式数据读取器
class CriteoReader : public DataReader {
public:
    CriteoReader(const String& file_path, Int batch_size, Int feature_dim);
    ~CriteoReader() override = default;
    
    // 获取下一批次数据
    bool NextBatch(Batch& batch) override;
    
    // 重置读取器到开始位置
    void Reset() override;
    
    // 读取一个样本
    bool ReadSample(Sample& sample) override;
    
    // 使用基类的Open方法
    bool Open(const String& file_path) override {
        return DataReader::Open(file_path);
    }
    
private:
    std::mutex mutex_;
};

// 数据格式转换辅助函数
DataFormat ParseDataFormat(const String& format_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_DATA_READER_H 