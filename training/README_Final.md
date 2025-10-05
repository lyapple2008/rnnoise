# RNNoise PyTorch训练和部署完整指南

本指南提供了从PyTorch训练到C推理的完整流程，基于原始的Keras实现但使用PyTorch框架。

## 完整流程概览

```
训练数据 → PyTorch训练 → 模型转换 → C推理
    ↓           ↓           ↓        ↓
denoise.c → weights.pth → rnn_data.c → RNNoise
```

## 1. 数据准备

### 生成训练数据

```bash
# 编译denoise.c（已修改支持直接文件输出）
cd src
gcc -DTRAINING=1 -O3 denoise.c celt_lpc.c kiss_fft.c pitch.c rnn.c rnn_data.c -o denoise_training

# 生成训练数据
./denoise_training speech.raw noise.raw 100000 > training_data.bin
```

### 验证数据格式

```bash
# 测试二进制数据格式
python3 test_binary_format.py --binary-file training_data.bin
```

## 2. PyTorch训练

### 基本训练

```bash
# 使用默认参数训练
python3 rnn_train_pytorch_new.py --data-file training_data.bin
```

### 自定义训练

```bash
# 自定义参数训练
python3 rnn_train_pytorch_new.py \
    --data-file training_data.bin \
    --epochs 120 \
    --batch-size 32 \
    --lr 0.001 \
    --device cpu \
    --output-model my_model.pth
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-file` | `training_10000.f32` | 训练数据文件 |
| `--epochs` | `120` | 训练轮数 |
| `--batch-size` | `32` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--device` | `auto` | 设备选择 |
| `--validation-split` | `0.1` | 验证集比例 |

## 3. 模型转换

### 自动转换

```bash
# 一键转换（推荐）
python3 convert_model.py
```

### 手动转换

```bash
# 手动指定参数
python3 dump_pytorch_model_fixed.py \
    weights.pth \
    rnnoise_weights.c \
    rnnoise_weights.bin \
    --model-name pytorch
```

## 4. C项目集成

### 复制文件

```bash
# 将生成的文件复制到源码目录
cp rnnoise_weights.c src/
cp rnnoise_weights.bin src/
```

### 修改源码

在`src/rnn_data.c`中添加：

```c
#include "rnnoise_weights.c"

// 使用新模型
extern const struct RNNModel rnnoise_model_pytorch;
```

### 编译项目

```bash
cd src
gcc -o rnnoise_demo rnnoise_demo.c denoise.c rnn.c rnnoise_weights.c
```

## 5. 验证和测试

### 功能测试

```bash
# 使用新模型进行推理
./rnnoise_demo input.wav output.wav
```

### 性能对比

```bash
# 对比原始模型和新模型的性能
time ./rnnoise_demo input.wav output_original.wav
time ./rnnoise_demo input.wav output_pytorch.wav
```

## 文件结构

```
training/
├── rnn_train_pytorch_new.py      # PyTorch训练脚本
├── dump_pytorch_model_fixed.py   # 模型转换脚本
├── convert_model.py              # 简化转换脚本
├── test_binary_format.py         # 数据格式测试
├── README_Final.md               # 本指南
└── README_Model_Conversion.md    # 详细转换说明

src/
├── denoise.c                     # 修改后的训练数据生成
├── rnn_data.c                    # 原始模型数据
├── rnn_data.h                    # 模型结构定义
├── rnnoise_weights.c            # 生成的PyTorch模型
└── rnnoise_weights.bin           # 二进制权重文件
```

## 关键改进

### 1. 解决Windows兼容性问题

- 修改`denoise.c`直接写入文件而不是stdout
- 避免Windows重定向的文本模式转换问题

### 2. 精确的Keras等价性

- 完全匹配原始Keras模型架构
- 实现相同的损失函数和权重约束
- 保持相同的训练行为

### 3. 简化的部署流程

- 一键转换脚本
- 自动文件管理
- 详细的错误提示

## 常见问题解决

### 1. 验证损失为0

**原因**: 数据量太少，验证集为空
**解决**: 增加训练数据量或调整验证分割比例

```bash
# 生成更多数据
./denoise_training speech.raw noise.raw 100000 > training_large.bin

# 或调整验证比例
python3 rnn_train_pytorch_new.py --validation-split 0.2
```

### 2. MPS设备错误

**原因**: Apple Silicon的MPS后端兼容性问题
**解决**: 使用CPU训练

```bash
python3 rnn_train_pytorch_new.py --device cpu
```

### 3. 内存不足

**原因**: 批次大小太大
**解决**: 减少批次大小

```bash
python3 rnn_train_pytorch_new.py --batch-size 16
```

### 4. 转换失败

**原因**: 模型文件损坏或路径错误
**解决**: 检查文件路径和权限

```bash
# 检查模型文件
ls -la weights.pth

# 重新训练
python3 rnn_train_pytorch_new.py --epochs 5 --output-model new_weights.pth
```

## 性能优化建议

### 训练优化

```bash
# 快速训练（较少数据）
python3 rnn_train_pytorch_new.py \
    --epochs 50 \
    --batch-size 64 \
    --window-size 1000

# 高质量训练（更多数据）
python3 rnn_train_pytorch_new.py \
    --epochs 200 \
    --batch-size 16 \
    --window-size 2000
```

### 推理优化

- 使用量化权重（已自动应用）
- 优化内存访问模式
- 利用SIMD指令集

## 总结

这个完整的流程提供了：

1. **跨平台兼容**: 解决Windows二进制数据问题
2. **精确等价**: 与原始Keras实现完全一致
3. **简化部署**: 一键转换和集成
4. **易于维护**: 清晰的代码结构和文档

通过这个流程，你可以：
- 使用PyTorch进行现代化训练
- 生成与原始RNNoise兼容的C模型
- 在任意平台上部署推理

整个过程保持了与原始实现的完全兼容性，同时提供了更好的开发体验和跨平台支持。
