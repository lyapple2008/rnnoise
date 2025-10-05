# PyTorch模型转换为C格式

本指南说明如何将训练好的PyTorch模型转换为C兼容的格式，用于RNNoise推理。

## 转换流程

### 1. 训练PyTorch模型

首先使用PyTorch训练脚本训练模型：

```bash
# 训练模型
python3 rnn_train_pytorch_new.py \
    --data-file training_data.bin \
    --epochs 120 \
    --batch-size 32 \
    --output-model weights.pth
```

### 2. 转换为C格式

使用转换脚本将PyTorch模型转换为C格式：

```bash
python3 dump_pytorch_model_fixed.py \
    weights.pth \
    rnnoise_weights.c \
    rnnoise_weights.bin \
    --model-name pytorch
```

### 3. 生成的文件

转换后会生成两个文件：

- **`rnnoise_weights.c`**: C源文件，包含量化的权重和模型结构
- **`rnnoise_weights.bin`**: 二进制权重文件，用于模型加载

## 文件格式说明

### C源文件格式

生成的`rnnoise_weights.c`文件包含：

1. **权重数组**: 量化的权重数据（-127到127的整数）
2. **层结构定义**: 每层的配置和权重指针
3. **模型结构**: 完整的RNNModel结构定义

```c
// 示例结构
const struct RNNModel rnnoise_model_pytorch = {
  24,                    // input_dense_size
  &input_dense,          // input_dense
  24,                    // vad_gru_size  
  &vad_gru,              // vad_gru
  48,                    // noise_gru_size
  &noise_gru,            // noise_gru
  96,                    // denoise_gru_size
  &denoise_gru,          // denoise_gru
  22,                    // denoise_output_size
  &denoise_output,       // denoise_output
  1,                     // vad_output_size
  &vad_output            // vad_output
};
```

### 二进制文件格式

二进制文件包含：
- 文件头：`rnnoise-nu model file version 1`
- 每层的配置：输入大小、输出大小、激活函数
- 量化的权重数据

## 集成到C项目

### 1. 复制生成的文件

将生成的文件复制到RNNoise源码目录：

```bash
cp rnnoise_weights.c src/
cp rnnoise_weights.bin src/
```

### 2. 修改rnn_data.c

在`src/rnn_data.c`中添加对新模型的支持：

```c
#include "rnnoise_weights.c"

// 添加模型引用
extern const struct RNNModel rnnoise_model_pytorch;
```

### 3. 修改rnn_data.h

在`src/rnn_data.h`中声明新模型：

```c
extern const struct RNNModel rnnoise_model_pytorch;
```

### 4. 使用新模型

在代码中使用新模型：

```c
// 创建RNNoise状态时指定模型
DenoiseState *st = rnnoise_create(&rnnoise_model_pytorch);
```

## 验证转换结果

### 1. 检查文件大小

```bash
# 检查生成的文件
ls -la rnnoise_weights.*

# 预期大小：
# rnnoise_weights.c: ~300KB
# rnnoise_weights.bin: ~270KB
```

### 2. 编译测试

```bash
# 编译RNNoise
cd src
gcc -c rnnoise_weights.c
gcc -c rnn.c
gcc -c denoise.c
# ... 其他源文件
```

### 3. 功能测试

```bash
# 使用新模型进行推理测试
./rnnoise_demo input.wav output.wav
```

## 权重量化说明

### 量化过程

1. **浮点权重**: PyTorch模型使用32位浮点权重
2. **量化**: 权重乘以256并四舍五入到整数
3. **范围限制**: 限制在-127到127之间
4. **存储**: 使用8位有符号整数存储

### 量化公式

```c
quantized_weight = min(127, max(-127, round(256 * float_weight)))
```

### 反量化

在C代码中使用时：

```c
float_weight = quantized_weight / 256.0f
```

## 模型结构对应关系

| PyTorch层 | C结构 | 说明 |
|-----------|-------|------|
| input_dense | DenseLayer | 输入密集层 |
| vad_gru | GRULayer | VAD GRU层 |
| noise_gru | GRULayer | 噪声GRU层 |
| denoise_gru | GRULayer | 去噪GRU层 |
| denoise_output | DenseLayer | 去噪输出层 |
| vad_output | DenseLayer | VAD输出层 |

## 激活函数映射

| PyTorch激活 | C常量 | 值 |
|-------------|-------|-----|
| tanh | ACTIVATION_TANH | 0 |
| sigmoid | ACTIVATION_SIGMOID | 1 |
| relu | ACTIVATION_RELU | 2 |

## 常见问题

### 1. 编译错误

**问题**: 找不到权重数组
**解决**: 确保`rnnoise_weights.c`被正确包含

### 2. 模型不匹配

**问题**: 模型结构不匹配
**解决**: 检查PyTorch模型架构是否与原始Keras模型一致

### 3. 权重精度损失

**问题**: 量化导致精度损失
**解决**: 这是正常的，量化是必要的优化

### 4. 内存使用

**问题**: 模型太大
**解决**: 检查权重数量，确保在合理范围内

## 性能优化

### 1. 权重压缩

量化后的权重比原始浮点权重小4倍：
- 原始: 32位浮点 = 4字节/权重
- 量化: 8位整数 = 1字节/权重

### 2. 内存访问

量化权重提高缓存效率，减少内存带宽需求。

### 3. 计算优化

整数运算比浮点运算更快，特别是在嵌入式设备上。

## 完整示例

```bash
# 1. 训练模型
python3 rnn_train_pytorch_new.py \
    --data-file training_data.bin \
    --epochs 120 \
    --output-model my_model.pth

# 2. 转换为C格式
python3 dump_pytorch_model_fixed.py \
    my_model.pth \
    my_model_weights.c \
    my_model_weights.bin \
    --model-name my_model

# 3. 集成到项目
cp my_model_weights.c src/
cp my_model_weights.bin src/

# 4. 编译和测试
cd src
gcc -o rnnoise_demo rnnoise_demo.c denoise.c rnn.c my_model_weights.c
./rnnoise_demo input.wav output.wav
```

这样就完成了从PyTorch训练到C推理的完整流程。
