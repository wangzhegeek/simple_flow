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
    virtual ~DataReader();

    // 获取下一批次数据
    virtual bool NextBatch(Batch& batch) = 0;
    
    // 重置读取器到开始位置
    virtual void Reset() = 0;
    
    // 获取特征维度
    Int GetFeatureDim() const { return feature_dim_; }
    
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
};

// LibSVM格式数据读取器
class LibSVMReader : public DataReader {
public:
    LibSVMReader(const String& file_path, Int batch_size, Int feature_dim);
    ~LibSVMReader() override;

    bool NextBatch(Batch& batch) override;
    void Reset() override;

private:
    std::mutex mutex_;
};

// Criteo格式数据读取器
class CriteoReader : public DataReader {
public:
    CriteoReader(const String& file_path, Int batch_size, Int feature_dim);
    ~CriteoReader() override;

    bool NextBatch(Batch& batch) override;
    void Reset() override;

private:
    std::mutex mutex_;
};

// 数据格式转换辅助函数
DataFormat ParseDataFormat(const String& format_str);

} // namespace simpleflow

#endif // SIMPLEFLOW_DATA_READER_H 