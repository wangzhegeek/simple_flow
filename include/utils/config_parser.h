#ifndef SIMPLEFLOW_UTILS_CONFIG_PARSER_H
#define SIMPLEFLOW_UTILS_CONFIG_PARSER_H

#include "types.h"
#include <unordered_map>
#include <fstream>
#include <iostream>

namespace simpleflow {
namespace utils {

// 配置解析器类
class ConfigParser {
public:
    ConfigParser();
    
    // 从文件解析配置
    bool ParseFile(const String& file_path);
    
    // 获取字符串配置项
    String GetString(const String& key, const String& default_value = "") const;
    
    // 获取整数配置项
    Int GetInt(const String& key, Int default_value = 0) const;
    
    // 获取浮点数配置项
    Float GetFloat(const String& key, Float default_value = 0.0) const;
    
    // 获取布尔配置项
    bool GetBool(const String& key, bool default_value = false) const;
    
    // 检查是否包含配置项
    bool Contains(const String& key) const;
    
    // 获取所有配置项
    const std::unordered_map<String, String>& GetAll() const;
    
    // 打印所有配置参数
    void PrintAllParameters() const {
        for (const auto& param : configs_) {
            std::cout << param.first << " = " << param.second << std::endl;
        }
    }
    
private:
    std::unordered_map<String, String> configs_;
    
    // 处理字符串，去除前后空格
    static String Trim(const String& str);
};

} // namespace utils
} // namespace simpleflow

#endif // SIMPLEFLOW_UTILS_CONFIG_PARSER_H 