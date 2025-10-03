# RNNoise PyTorch Training

This directory contains a PyTorch-based implementation of the RNNoise training pipeline, equivalent to the original Keras version but using modern PyTorch.

## Files

- **`rnn_train_pytorch.py`**: Main PyTorch training script
- **`dump_pytorch_model.py`**: Converts trained PyTorch models to C-compatible format  
- **`train_example.py`**: Example script showing usage
- **`README_PyTorch.md`**: This documentation file

## Requirements

```bash
pip install torch numpy h5py argparse
```

## Model Architecture

The PyTorch implementation recreates the exact same architecture as the original Keras model:

1. **Input Dense Layer**: 42 → 24 neurons, tanh activation
2. **VAD GRU**: 24 hidden units, tanh/sigmoid activations
3. **Noise GRU**: 48 hidden units, ReLU/sigmoid activations (input: concat of dense_out + vad_out + main_input = 90)
4. **Denoise GRU**: 96 hidden units, tanh/sigmoid activations (input: concat of vad_out + noise_out + main_input = 114)
5. **Denoise Output**: 96 → 22 neurons, sigmoid activation
6. **VAD Output**: 24 → 1 neuron, sigmoid activation

## Usage

### Basic Training

```bash
# Train with default parameters
python3 rnn_train_pytorch.py --data-file training.h5

# Train with custom parameters  
python3 rnn_train_pytorch.py \
    --data-file training.h5 \
    --batch-size 32 \
    --epochs 120 \
    --lr 0.001 \
    --output-model my_rnnoise.pth
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-file` | `training.h5` | Path to HDF5 training data |
| `--batch-size` | `32` | Batch size for training |
| `--epochs` | `120` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--validation-split` | `0.1` | Fraction for validation |
| `--window-size` | `2000` | Sequence length |
| `--output-model` | `rnnoise_pytorch.pth` | Output model filename |
| `--device` | `auto` | Device: auto/cpu/cuda/mps |
| `--l2-reg` | `1e-6` | L2 regularization |
| `--constraint-clip` | `0.499` | Weight clipping value |
| `--denoise-weight` | `10.0` | Loss weight for denoise output |
| `--vad-weight` | `0.5` | Loss weight for VAD output |

### Data Format

The training data should be an HDF5 file with a dataset named `'data'` containing:

- **Shape**: `(N, 87)` where N is the number of samples
- **Columns 0-41**: Input features (42 dimensions)
- **Columns 42-63**: Denoise targets (22 dimensions)
- **Columns 64-85**: Noise targets (22 dimensions, currently unused)
- **Column 86**: VAD targets (1 dimension)

### Model Conversion

Convert the trained PyTorch model to C-compatible format:

```bash
python3 dump_pytorch_model.py \
    rnnoise_pytorch.pth \
    rnnoise_weights.c \
    rnnoise_weights.bin \
    --model-name my_model
```

This generates:
- **C source file**: Contains quantized weights and model structure
- **Binary weights file**: For use with the C model loader

## Key Features

### Loss Functions

The implementation includes exact PyTorch equivalents of the original Keras loss functions:

- **Custom Denoise Loss** (`mycost`): Complex loss combining squared differences and binary cross-entropy
- **Custom VAD Cross-entropy** (`my_crossentropy`): Weighted binary cross-entropy  
- **MSSE Loss** (`msse`): Mean Squared Square-root Error for metrics

### Weight Constraints

Implements weight clipping constraints equivalent to Keras `WeightClip`:
- Clips all weights to `[-0.499, 0.499]` range
- Applied before and after each optimizer step
- Ensures compatibility with quantized C implementation

### Model Architecture Fidelity

- **Exact layer dimensions**: Matches original Keras model
- **Activation functions**: Proper tanh/sigmoid/ReLU placement
- **GRU implementations**: Standard PyTorch GRU with custom activations applied
- **Concatenation logic**: Recreates complex input concatenations

## Example Training Session

```bash
# Run example training (10 epochs for testing)
python3 train_example.py

# Full training session
python3 rnn_train_pytorch.py \
    --data-file training.h5 \
    --epochs 120 \
    --batch-size 32 \
    --output-model rnnoise_full.pth

# Convert to C format
python3 dump_pytorch_model.py \
    rnnoise_full.pth \
    rnnoise_full_weights.c \
    rnnoise_full_weights.bin \
    --model-name full
```

## GPU Support

The training script automatically detects and uses available GPU acceleration:

- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon Macs with Metal Performance Shaders
- **CPU**: Fallback for systems without GPU acceleration

Force a specific device:
```bash
python3 rnn_train_pytorch.py --device cuda    # Force CUDA
python3 rnn_train_pytorch.py --device mps     # Force MPS (Apple Silicon)
python3 rnn_train_pytorch.py --device cpu     # Force CPU
```

## Differences from Keras Version

While functionally equivalent, the PyTorch version offers several advantages:

1. **Modern Framework**: Uses current PyTorch (vs older Keras/TensorFlow)
2. **Better GPU Support**: Improved CUDA/MPS support
3. **Flexible Training**: More customizable training loops
4. **Better Debugging**: Easier to inspect and modify model behavior
5. **Model Export**: Direct conversion to C-compatible format

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python3 rnn_train_pytorch.py --batch-size 16 --device cpu
   ```

2. **Missing Dependencies**:
   ```bash
   pip install torch numpy h5py
   ```

3. **Data Loading Errors**:
   - Ensure `training.h5` exists and has correct format
   - Check data shape is `(N, 87)`

4. **Slow Training**:
   - Use GPU if available: `--device cuda` or `--device mps`
   - Increase batch size: `--batch-size 64`
   - Reduce sequence length: `--window-size 1000`

### Performance Optimization

For faster training:
```bash
python3 rnn_train_pytorch.py \
    --batch-size 64 \
    --device cuda \
    --window-size 1000 \
    --validation-split 0.05
```

For better accuracy:
```bash
python3 rnn_train_pytorch.py \
    --batch-size 16 \
    --epochs 200 \
    --lr 0.0005 \
    --window-size 2000
```