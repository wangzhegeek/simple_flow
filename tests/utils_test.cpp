#include <gtest/gtest.h>
#include "simpleflow/utils/math_util.h"
#include "simpleflow/utils/config_parser.h"
#include <cmath>
#include <vector>
#include <fstream>

namespace simpleflow {
namespace utils {
namespace test {

class MathUtilTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化测试数据
        x_ = new Float[5]{1.0, 2.0, 3.0, 4.0, 5.0};
        y_ = new Float[5]{5.0, 4.0, 3.0, 2.0, 1.0};
        z_ = new Float[5];
    }
    
    void TearDown() override {
        delete[] x_;
        delete[] y_;
        delete[] z_;
    }
    
    Float* x_;
    Float* y_;
    Float* z_;
};

TEST_F(MathUtilTest, DotProduct) {
    Float result = MathUtil::DotProduct(x_, y_, 5);
    EXPECT_FLOAT_EQ(result, 35.0);
}

TEST_F(MathUtilTest, VectorAdd) {
    MathUtil::VectorAdd(x_, y_, z_, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], 6.0);
    }
}

TEST_F(MathUtilTest, VectorSubtract) {
    MathUtil::VectorSubtract(x_, y_, z_, 5);
    Float expected[5] = {-4.0, -2.0, 0.0, 2.0, 4.0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], expected[i]);
    }
}

TEST_F(MathUtilTest, VectorScale) {
    MathUtil::VectorScale(x_, 2.0, z_, 5);
    Float expected[5] = {2.0, 4.0, 6.0, 8.0, 10.0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], expected[i]);
    }
}

TEST_F(MathUtilTest, VectorAddScalar) {
    MathUtil::VectorAddScalar(x_, 1.0, z_, 5);
    Float expected[5] = {2.0, 3.0, 4.0, 5.0, 6.0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], expected[i]);
    }
}

TEST_F(MathUtilTest, VectorMultiply) {
    MathUtil::VectorMultiply(x_, y_, z_, 5);
    Float expected[5] = {5.0, 8.0, 9.0, 8.0, 5.0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], expected[i]);
    }
}

TEST_F(MathUtilTest, VectorSquare) {
    MathUtil::VectorSquare(x_, z_, 5);
    Float expected[5] = {1.0, 4.0, 9.0, 16.0, 25.0};
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(z_[i], expected[i]);
    }
}

TEST_F(MathUtilTest, VectorSquareSum) {
    Float result = MathUtil::VectorSquareSum(x_, 5);
    EXPECT_FLOAT_EQ(result, 55.0);
}

TEST_F(MathUtilTest, VectorSum) {
    Float result = MathUtil::VectorSum(x_, 5);
    EXPECT_FLOAT_EQ(result, 15.0);
}

TEST_F(MathUtilTest, Sigmoid) {
    Float result = MathUtil::Sigmoid(0.0);
    EXPECT_FLOAT_EQ(result, 0.5);
    
    Float values[3] = {-1.0, 0.0, 1.0};
    Float expected[3] = {
        1.0 / (1.0 + std::exp(1.0)),
        0.5,
        1.0 / (1.0 + std::exp(-1.0))
    };
    
    Float results[3];
    MathUtil::Sigmoid(values, results, 3);
    
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(results[i], expected[i]);
    }
}

TEST_F(MathUtilTest, LogLoss) {
    Float pred = 0.7;
    Float target = 1.0;
    Float result = MathUtil::LogLoss(pred, target);
    Float expected = -std::log(pred);
    EXPECT_FLOAT_EQ(result, expected);
    
    pred = 0.3;
    target = 0.0;
    result = MathUtil::LogLoss(pred, target);
    expected = -std::log(1.0 - pred);
    EXPECT_FLOAT_EQ(result, expected);
}

class ConfigParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试配置文件
        std::ofstream config_file("test_config.conf");
        config_file << "# 测试配置文件\n";
        config_file << "string_key = test_value\n";
        config_file << "int_key = 123\n";
        config_file << "float_key = 3.14\n";
        config_file << "bool_key1 = true\n";
        config_file << "bool_key2 = yes\n";
        config_file << "bool_key3 = 1\n";
        config_file.close();
    }
    
    void TearDown() override {
        // 删除测试配置文件
        std::remove("test_config.conf");
    }
    
    ConfigParser parser_;
};

TEST_F(ConfigParserTest, ParseFile) {
    EXPECT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_FALSE(parser_.ParseFile("non_existent_file.conf"));
}

TEST_F(ConfigParserTest, GetString) {
    ASSERT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_EQ(parser_.GetString("string_key"), "test_value");
    EXPECT_EQ(parser_.GetString("non_existent_key", "default"), "default");
}

TEST_F(ConfigParserTest, GetInt) {
    ASSERT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_EQ(parser_.GetInt("int_key"), 123);
    EXPECT_EQ(parser_.GetInt("non_existent_key", 456), 456);
    EXPECT_EQ(parser_.GetInt("string_key", 789), 789);  // 无法解析为int，返回默认值
}

TEST_F(ConfigParserTest, GetFloat) {
    ASSERT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_FLOAT_EQ(parser_.GetFloat("float_key"), 3.14f);
    EXPECT_FLOAT_EQ(parser_.GetFloat("non_existent_key", 2.71f), 2.71f);
    EXPECT_FLOAT_EQ(parser_.GetFloat("string_key", 1.23f), 1.23f);  // 无法解析为float，返回默认值
}

TEST_F(ConfigParserTest, GetBool) {
    ASSERT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_TRUE(parser_.GetBool("bool_key1"));
    EXPECT_TRUE(parser_.GetBool("bool_key2"));
    EXPECT_TRUE(parser_.GetBool("bool_key3"));
    EXPECT_FALSE(parser_.GetBool("non_existent_key"));
    EXPECT_FALSE(parser_.GetBool("string_key"));  // 不是bool值
}

TEST_F(ConfigParserTest, Contains) {
    ASSERT_TRUE(parser_.ParseFile("test_config.conf"));
    EXPECT_TRUE(parser_.Contains("string_key"));
    EXPECT_FALSE(parser_.Contains("non_existent_key"));
}

} // namespace test
} // namespace utils
} // namespace simpleflow 