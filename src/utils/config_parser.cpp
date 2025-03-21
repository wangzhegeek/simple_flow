#include "utils/config_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace simpleflow {
namespace utils {

ConfigParser::ConfigParser() {}

bool ConfigParser::ParseFile(const String& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    String line;
    while (std::getline(file, line)) {
        // 去除前后空格
        line = Trim(line);
        
        // 跳过空行和注释行
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // 解析键值对
        size_t pos = line.find('=');
        if (pos != String::npos) {
            String key = Trim(line.substr(0, pos));
            String value = Trim(line.substr(pos + 1));
            
            if (!key.empty()) {
                configs_[key] = value;
            }
        }
    }
    
    return true;
}

String ConfigParser::GetString(const String& key, const String& default_value) const {
    auto it = configs_.find(key);
    return (it != configs_.end()) ? it->second : default_value;
}

Int ConfigParser::GetInt(const String& key, Int default_value) const {
    auto it = configs_.find(key);
    if (it != configs_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

Float ConfigParser::GetFloat(const String& key, Float default_value) const {
    auto it = configs_.find(key);
    if (it != configs_.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

bool ConfigParser::GetBool(const String& key, bool default_value) const {
    auto it = configs_.find(key);
    if (it != configs_.end()) {
        String value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return (value == "true" || value == "yes" || value == "1");
    }
    return default_value;
}

bool ConfigParser::Contains(const String& key) const {
    return configs_.find(key) != configs_.end();
}

const std::unordered_map<String, String>& ConfigParser::GetAll() const {
    return configs_;
}

String ConfigParser::Trim(const String& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (first == String::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, last - first + 1);
}

} // namespace utils
} // namespace simpleflow 