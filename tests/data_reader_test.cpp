#include <gtest/gtest.h>
#include "data_reader.h"
#include "types.h"
#include <fstream>
#include <cstdio>
#include <memory>
#include <vector>

namespace simpleflow {
namespace test {

class DataReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建一个临时LibSVM格式文件用于测试
        test_file_ = "test_data.libsvm";
        std::ofstream file(test_file_);
        file << "0 0:1.0 1:2.0 2:3.0\n"
             << "1 0:4.0 1:5.0 2:6.0\n"
             << "0 0:7.0 1:8.0 2:9.0\n"
             << "1 0:10.0 1:11.0 2:12.0\n";
        file.close();
    }
    
    void TearDown() override {
        // 删除临时文件
        std::remove(test_file_.c_str());
    }
    
    std::string test_file_;
};

TEST_F(DataReaderTest, LibSVMReader) {
    // 创建LibSVM读取器
    LibSVMReader reader(test_file_, 2, 3);
    
    // 测试获取批次
    Batch batch;
    
    // 获取第一个批次
    EXPECT_TRUE(reader.NextBatch(batch));
    EXPECT_EQ(batch.size(), 2);
    
    // 验证第一个批次的内容
    ASSERT_EQ(batch[0].features.size(), 3);
    EXPECT_FLOAT_EQ(batch[0].features[0].value, 1.0);
    EXPECT_FLOAT_EQ(batch[0].features[1].value, 2.0);
    EXPECT_FLOAT_EQ(batch[0].features[2].value, 3.0);
    EXPECT_FLOAT_EQ(batch[0].label, 0.0);
    
    ASSERT_EQ(batch[1].features.size(), 3);
    EXPECT_FLOAT_EQ(batch[1].features[0].value, 4.0);
    EXPECT_FLOAT_EQ(batch[1].features[1].value, 5.0);
    EXPECT_FLOAT_EQ(batch[1].features[2].value, 6.0);
    EXPECT_FLOAT_EQ(batch[1].label, 1.0);
    
    // 获取第二个批次
    EXPECT_TRUE(reader.NextBatch(batch));
    EXPECT_EQ(batch.size(), 2);
    
    // 验证第二个批次的内容
    ASSERT_EQ(batch[0].features.size(), 3);
    EXPECT_FLOAT_EQ(batch[0].features[0].value, 7.0);
    EXPECT_FLOAT_EQ(batch[0].features[1].value, 8.0);
    EXPECT_FLOAT_EQ(batch[0].features[2].value, 9.0);
    EXPECT_FLOAT_EQ(batch[0].label, 0.0);
    
    ASSERT_EQ(batch[1].features.size(), 3);
    EXPECT_FLOAT_EQ(batch[1].features[0].value, 10.0);
    EXPECT_FLOAT_EQ(batch[1].features[1].value, 11.0);
    EXPECT_FLOAT_EQ(batch[1].features[2].value, 12.0);
    EXPECT_FLOAT_EQ(batch[1].label, 1.0);
}

TEST_F(DataReaderTest, LibSVMReaderReset) {
    LibSVMReader reader(test_file_, 2, 3);
    
    // 测试重置
    Batch batch1, batch2;
    
    reader.NextBatch(batch1);
    reader.Reset();
    reader.NextBatch(batch2);
    
    // 重置后，应该能够获取相同的数据
    ASSERT_EQ(batch1.size(), batch2.size());
    
    for (size_t i = 0; i < batch1.size(); ++i) {
        ASSERT_EQ(batch1[i].features.size(), batch2[i].features.size());
        for (size_t j = 0; j < batch1[i].features.size(); ++j) {
            EXPECT_FLOAT_EQ(batch1[i].features[j].value, batch2[i].features[j].value);
        }
        EXPECT_FLOAT_EQ(batch1[i].label, batch2[i].label);
    }
}

TEST_F(DataReaderTest, CriteoReader) {
    // 创建Criteo格式的测试文件
    std::string criteo_file = "test_criteo.txt";
    std::ofstream file(criteo_file);
    file << "0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\ta\tb\tc\td\te\tf\tg\th\ti\tj\tk\tl\tm\tn\to\tp\tq\tr\ts\tt\tu\tv\tw\tx\ty\tz\n"
         << "1\t14\t15\t16\t17\t18\t19\t20\t21\t22\t23\t24\t25\t26\taa\tbb\tcc\tdd\tee\tff\tgg\thh\tii\tjj\tkk\tll\tmm\tnn\too\tpp\tqq\trr\tss\ttt\tuu\tvv\tww\txx\tyy\tzz\n";
    file.close();
    
    // 创建Criteo读取器
    CriteoReader reader(criteo_file, 1, 100);
    
    // 测试获取批次
    Batch batch;
    
    // 获取批次
    EXPECT_TRUE(reader.NextBatch(batch));
    EXPECT_EQ(batch.size(), 1);
    
    // 验证特征数量（应该有各种特征，但数量可能会根据实际实现而有所不同）
    EXPECT_GT(batch[0].features.size(), 0);
    
    // 清理测试文件
    std::remove(criteo_file.c_str());
}

TEST_F(DataReaderTest, LibSVMReaderInvalidFile) {
    // 测试打开不存在的文件
    EXPECT_THROW(LibSVMReader("nonexistent_file.libsvm", 2, 3), std::runtime_error);
}

TEST_F(DataReaderTest, Create) {
    // 测试创建不同类型的数据读取器
    std::shared_ptr<DataReader> reader = DataReader::Create(
        DataFormat::LIBSVM, test_file_, 2, 3);
    
    // 测试获取特征维度
    EXPECT_EQ(reader->GetFeatureDim(), 3);
    
    // 测试创建未知类型的数据读取器
    EXPECT_THROW(DataReader::Create(DataFormat::Unknown, 
                                   test_file_, 2, 3), 
                std::runtime_error);
}

TEST_F(DataReaderTest, ParseDataFormat) {
    EXPECT_EQ(ParseDataFormat("libsvm"), DataFormat::LIBSVM);
    EXPECT_EQ(ParseDataFormat("criteo"), DataFormat::CRITEO);
    EXPECT_EQ(ParseDataFormat("csv"), DataFormat::CSV);
    EXPECT_EQ(ParseDataFormat("unknown"), DataFormat::Unknown);
}

} // namespace test
} // namespace simpleflow 