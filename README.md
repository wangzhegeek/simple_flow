# SimpleFlow: 轻量级推荐算法框架

SimpleFlow是一个用C++实现的轻量级推荐算法框架，专注于大规模稀疏特征下的推荐任务，如CTR预估等。

## 特性

- 高效多线程训练，支持CPU并行计算
- 模块化设计，易于扩展
- 支持常用的推荐模型：LR、FM
- 支持常用的数据格式：LibSVM、Criteo
- 使用BLAS库优化底层矩阵运算
- 可通过配置文件设置超参数
- FM模型支持高效的因子分解计算和数值稳定性优化
- 自适应学习率调整机制
- 线程安全的模型更新

## 目录结构

```
├── include/            # 头文件
│   ├── activation.h    # 激活函数
│   ├── data_reader.h   # 数据读取
│   ├── loss.h          # 损失函数
│   ├── metric.h        # 评估指标
│   ├── model.h         # 模型基类
│   ├── optimizer.h     # 优化器
│   ├── trainer.h       # 训练器
│   ├── types.h         # 类型定义
│   ├── utils.h         # 工具函数
│   ├── models/         # 具体模型定义
│   └── utils/          # 工具类
├── src/                # 源代码
│   ├── models/         # 模型实现
│   └── utils/          # 工具类实现
├── examples/           # 示例代码
├── tests/              # 单元测试
├── config/             # 配置文件
├── model/              # 模型存储
└── data/               # 示例数据
```

## 依赖

- C++14或更高版本
- BLAS库（如OpenBLAS、MKL等）
- CMake 3.10+
- 支持pthread的系统环境

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

### LR模型配置

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
num_threads = 8           # 使用8个线程进行并行训练
verbose = true
log_interval = 100        # 每100个批次打印一次日志
model_save_path = ./model/model.bin
lr_decay_epochs = 3       # 每3轮降低学习率
lr_decay_factor = 0.8     # 学习率衰减因子
min_learning_rate = 1e-6  # 最小学习率
```

### FM模型配置

```ini
# 数据配置
data_format = libsvm
train_data_path = ./data/train.txt
test_data_path = ./data/test.txt
feature_dim = 1000000

# 模型配置
model_type = fm
embedding_size = 8
learning_rate = 0.01
batch_size = 128
epochs = 10
l2_reg = 0.0001

# 训练配置
num_threads = 8           # 使用8个线程进行并行训练
verbose = true
log_interval = 100        # 每100个批次打印一次日志
model_save_path = ./model/fm_model.bin
lr_decay_epochs = 3       # 每3轮降低学习率
lr_decay_factor = 0.8     # 学习率衰减因子
min_learning_rate = 1e-6  # 最小学习率
```

## 多线程训练

SimpleFlow实现了高效的多线程训练机制，主要特点包括：

1. **线程池实现**：使用C++标准库实现线程池，避免频繁创建和销毁线程的开销
2. **批次并行处理**：同时对多个数据批次进行并行训练，充分利用多核处理器
3. **线程安全机制**：
   - 使用互斥锁保护模型参数的并发访问
   - 批次结果安全传递和处理
   - 避免竞态条件导致的不确定性
4. **动态负载均衡**：根据可用线程数自动调整并行度
5. **性能优化**：
   - 减少线程同步开销
   - 避免"忙等待"造成的CPU资源浪费
   - 优化内存使用，减少不必要的数据复制

### 配置多线程训练

在配置文件中设置`num_threads`参数即可控制训练使用的线程数：
```ini
num_threads = 8  # 使用8线程并行训练
```

## FM模型优化

FM模型实现了以下优化：

1. 使用标准公式计算二阶特征交互: `0.5 * sum_f[(sum_i v_i,f * x_i)^2 - sum_i (v_i,f * x_i)^2]`
2. 采用延迟初始化策略，节省内存
3. 增强数值稳定性处理，避免梯度爆炸问题
4. 多级梯度裁剪和参数约束
5. 支持多线程并行训练

## 自适应学习率调整

框架支持自动检测和处理训练过程中的不稳定性：

1. **梯度爆炸检测**：监控损失值是否出现NaN或Inf
2. **异常梯度处理**：自动调整学习率或跳过问题批次
3. **学习率衰减策略**：
   - 定期衰减：每隔`lr_decay_epochs`轮次，将学习率乘以`lr_decay_factor`
   - 自适应衰减：当损失突然增大时，自动降低学习率
   - 最小学习率保障：确保学习率不低于`min_learning_rate`

## 框架扩展

### 添加新模型

1. 在`include/models/`目录下创建新模型的头文件
2. 在`src/models/`目录下实现模型
3. 在模型工厂方法中注册新模型

### 添加新的数据格式

1. 在`include/data_reader.h`中添加新的数据格式类型
2. 实现对应的数据读取器类
3. 在数据读取器工厂方法中注册新格式

## 许可证

MIT 