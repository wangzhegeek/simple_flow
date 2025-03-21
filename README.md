# SimpleFlow: 轻量级推荐算法框架

SimpleFlow是一个用C++实现的轻量级推荐算法框架，专注于大规模稀疏特征下的推荐任务，如CTR预估等。

## 特性

- 支持多线程CPU训练
- 模块化设计，易于扩展
- 支持常用的推荐模型：LR、FM
- 支持常用的数据格式：LibSVM、Criteo
- 使用BLAS库优化底层矩阵运算
- 可通过配置文件设置超参数

## 目录结构

```
├── include/            # 头文件
│   └── simpleflow/     # 框架头文件
│       ├── models/     # 模型定义
│       └── utils/      # 工具类
├── src/                # 源代码
│   ├── models/         # 模型实现
│   └── utils/          # 工具类实现
├── examples/           # 示例代码
├── tests/              # 单元测试
├── config/             # 配置文件
└── data/               # 示例数据
```

## 依赖

- C++14或更高版本
- BLAS库（如OpenBLAS、MKL等）
- CMake 3.10+

## 编译

```bash
mkdir build
cd build
cmake ..
make -j4
```

## 使用示例

### 训练LR模型

```bash
./build/examples/lr_example ./config/lr_train.conf
```

### 训练FM模型

```bash
./build/examples/fm_example ./config/fm_train.conf
```

## 配置文件示例

```ini
# 数据配置
data_format = libsvm
train_data_path = ./data/train.txt
test_data_path = ./data/test.txt
feature_dim = 1000000

# 模型配置
model_type = lr
learning_rate = 0.01
batch_size = 128
epochs = 10
l2_reg = 0.001

# 训练配置
num_threads = 8
verbose = true
model_save_path = ./model/model.bin
```

## 框架扩展

### 添加新模型

1. 在`include/simpleflow/models/`目录下创建新模型的头文件
2. 在`src/models/`目录下实现模型
3. 在模型工厂方法中注册新模型

### 添加新的数据格式

1. 在`include/simpleflow/data_reader.h`中添加新的数据格式类型
2. 实现对应的数据读取器类
3. 在数据读取器工厂方法中注册新格式

## 许可证

MIT 