#!/usr/bin/env python3
"""
PyTorch-based training script for RNNoise
Equivalent to the original Keras implementation but using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import argparse
import os
from typing import Tuple, Optional


class WeightConstraint:
    """Weight clipping constraint equivalent to Keras WeightClip"""
    def __init__(self, clip_value: float = 0.499):
        self.clip_value = clip_value
    
    def __call__(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                module.weight.clamp_(-self.clip_value, self.clip_value)
        if hasattr(module, 'bias') and module.bias is not None and not isinstance(module.bias, bool):
            with torch.no_grad():
                module.bias.clamp_(-self.clip_value, self.clip_value)


class RNNoiseModel(nn.Module):
    """
    PyTorch implementation of RNNoise model
    
    Architecture:
    - Input Dense: 42 -> 24 (tanh)
    - VAD GRU: 24 hidden units (tanh/sigmoid)
    - Noise GRU: 48 hidden units (relu/sigmoid) 
    - Denoise GRU: 96 hidden units (tanh/sigmoid)
    - Denoise Output: -> 22 (sigmoid)
    - VAD Output: -> 1 (sigmoid)
    """
    
    def __init__(self, l2_reg: float = 1e-6, constraint_clip: float = 0.499):
        super(RNNoiseModel, self).__init__()
        
        self.constraint = WeightConstraint(constraint_clip)
        
        # Input dense layer: 42 -> 24
        self.input_dense = nn.Linear(42, 24)
        
        # VAD GRU: 24 hidden units
        self.vad_gru = nn.GRU(input_size=24, hidden_size=24, batch_first=True)
        
        # Noise GRU: 48 hidden units  
        # Input: concat(input_dense_out=24, vad_gru_out=24, main_input=42) = 90
        self.noise_gru = nn.GRU(input_size=90, hidden_size=48, batch_first=True)
        
        # Denoise GRU: 96 hidden units
        # Input: concat(vad_gru_out=24, noise_gru_out=48, main_input=42) = 114  
        self.denoise_gru = nn.GRU(input_size=114, hidden_size=96, batch_first=True)
        
        # Output layers
        self.denoise_output = nn.Linear(96, 22)
        self.vad_output = nn.Linear(24, 1)
        
        # Apply weight constraints
        self._apply_constraints()
        
    def _apply_constraints(self):
        """Apply weight constraints to all layers"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.GRU)):
                self.constraint(module)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 42)
            
        Returns:
            Tuple of (denoise_output, vad_output)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input dense layer with tanh activation
        dense_out = torch.tanh(self.input_dense(x))  # (B, T, 24)
        
        # VAD GRU with tanh activation and sigmoid recurrent activation
        vad_gru_out, _ = self.vad_gru(dense_out)  # (B, T, 24)
        
        # VAD output
        vad_out = torch.sigmoid(self.vad_output(vad_gru_out))  # (B, T, 1)
        
        # Prepare noise GRU input: concat(dense_out, vad_gru_out, x)
        noise_input = torch.cat([dense_out, vad_gru_out, x], dim=-1)  # (B, T, 90)
        
        # Noise GRU with ReLU activation and sigmoid recurrent activation  
        noise_gru_out, _ = self.noise_gru(noise_input)  # (B, T, 48)
        # Apply ReLU activation to hidden states
        noise_gru_out = F.relu(noise_gru_out)
        
        # Prepare denoise GRU input: concat(vad_gru_out, noise_gru_out, x)
        denoise_input = torch.cat([vad_gru_out, noise_gru_out, x], dim=-1)  # (B, T, 114)
        
        # Denoise GRU with tanh activation and sigmoid recurrent activation
        denoise_gru_out, _ = self.denoise_gru(denoise_input)  # (B, T, 96)
        
        # Denoise output with sigmoid activation
        denoise_out = torch.sigmoid(self.denoise_output(denoise_gru_out))  # (B, T, 22)
        
        return denoise_out, vad_out


class RNNoiseDataset(Dataset):
    """Dataset class for RNNoise training data"""
    
    def __init__(self, data_file: str, window_size: int = 2000, data_format: str = 'hdf5'):
        """
        Initialize dataset
        
        Args:
            data_file: Path to data file (HDF5 or binary format)
            window_size: Length of sequences to generate
            data_format: 'hdf5' for HDF5 format, 'binary' for denoise.c output format
        """
        self.window_size = window_size
        self.data_format = data_format
        
        if data_format == 'hdf5':
            self._load_hdf5_data(data_file)
        elif data_format == 'binary':
            self._load_binary_data(data_file)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _load_hdf5_data(self, data_file: str):
        """Load data from HDF5 format (original implementation)"""
        print(f'Loading HDF5 data from {data_file}...')
        with h5py.File(data_file, 'r') as hf:
            all_data = hf['data'][:]
        print('HDF5 data loading complete.')
        
        # Calculate number of sequences
        self.nb_sequences = len(all_data) // self.window_size
        print(f'{self.nb_sequences} sequences of length {self.window_size}')
        
        # Reshape data into sequences
        data_truncated = all_data[:self.nb_sequences * self.window_size]
        
        # Input features (first 42 columns)
        self.x_data = data_truncated[:, :42].reshape(self.nb_sequences, self.window_size, 42)
        
        # Denoise targets (columns 42:64)
        self.y_denoise = data_truncated[:, 42:64].reshape(self.nb_sequences, self.window_size, 22)
        
        # Noise targets (columns 64:86) - not used in current loss but kept for compatibility
        self.noise_data = data_truncated[:, 64:86].reshape(self.nb_sequences, self.window_size, 22)
        
        # VAD targets (column 86)
        self.y_vad = data_truncated[:, 86:87].reshape(self.nb_sequences, self.window_size, 1)
        
        # Convert to float32
        self.x_data = self.x_data.astype(np.float32)
        self.y_denoise = self.y_denoise.astype(np.float32)
        self.y_vad = self.y_vad.astype(np.float32)
        
        print(f'HDF5 data shapes: x={self.x_data.shape}, y_denoise={self.y_denoise.shape}, y_vad={self.y_vad.shape}')
    
    def _load_binary_data(self, data_file: str):
        """Load data from binary format (denoise.c output)"""
        print(f'Loading binary data from {data_file}...')
        
        # Each frame contains: 42 (features) + 22 (denoise_targets) + 22 (noise_targets) + 1 (vad) = 87 floats
        frame_size = 87
        
        # Read binary data
        with open(data_file, 'rb') as f:
            data_bytes = f.read()
        
        # Convert to numpy array of floats
        total_floats = len(data_bytes) // 4  # 4 bytes per float
        if total_floats % frame_size != 0:
            print(f"Warning: Data size ({total_floats} floats) is not divisible by frame size ({frame_size}).")
            # Truncate to nearest complete frame
            total_floats = (total_floats // frame_size) * frame_size
        
        # Load as float32 array
        all_data = np.frombuffer(data_bytes[:total_floats*4], dtype=np.float32)
        total_frames = total_floats // frame_size
        
        print(f'Loaded {total_frames} frames from binary file')
        
        # Reshape into frames
        all_data = all_data.reshape(total_frames, frame_size)
        
        # Calculate number of sequences
        self.nb_sequences = total_frames // self.window_size
        print(f'{self.nb_sequences} sequences of length {self.window_size}')
        
        # Truncate to complete sequences
        data_truncated = all_data[:self.nb_sequences * self.window_size]
        
        # Extract different components
        # Features: columns 0:42
        self.x_data = data_truncated[:, :42].reshape(self.nb_sequences, self.window_size, 42)
        
        # Denoise targets: columns 42:64 (frequency band gains)
        self.y_denoise = data_truncated[:, 42:64].reshape(self.nb_sequences, self.window_size, 22)
        
        # Noise targets: columns 64:86 (log bark band energy) - not used in current loss
        self.noise_data = data_truncated[:, 64:86].reshape(self.nb_sequences, self.window_size, 22)
        
        # VAD targets: column 86
        self.y_vad = data_truncated[:, 86:87].reshape(self.nb_sequences, self.window_size, 1)
        
        # Convert to float32 (already float32 from binary, but ensure consistency)
        self.x_data = self.x_data.astype(np.float32)
        self.y_denoise = self.y_denoise.astype(np.float32)
        self.y_vad = self.y_vad.astype(np.float32)
        
        print(f'Binary data shapes: x={self.x_data.shape}, y_denoise={self.y_denoise.shape}, y_vad={self.y_vad.shape}')        
        print(f'Sample ranges:')
        print(f'  Features: [{np.min(self.x_data):.3f}, {np.max(self.x_data):.3f}]')
        print(f'  Denoise targets: [{np.min(self.y_denoise):.3f}, {np.max(self.y_denoise):.3f}]')
        print(f'  VAD targets: [{np.min(self.y_vad):.3f}, {np.max(self.y_vad):.3f}]')
    
    def __len__(self):
        return self.nb_sequences
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_data[idx]),
            torch.from_numpy(self.y_denoise[idx]),
            torch.from_numpy(self.y_vad[idx])
        )


def create_mask(y_true: torch.Tensor) -> torch.Tensor:
    """Create mask equivalent to Keras mymask function"""
    return torch.minimum(y_true + 1.0, torch.ones_like(y_true))


def msse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Squared Square-root Error loss equivalent to Keras msse"""
    mask = create_mask(y_true)
    sqrt_true = torch.sqrt(torch.clamp(y_true, min=1e-8))
    sqrt_pred = torch.sqrt(torch.clamp(y_pred, min=1e-8))
    return torch.mean(mask * torch.square(sqrt_pred - sqrt_true))


def custom_denoise_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Custom denoise loss equivalent to Keras mycost function"""
    mask = create_mask(y_true)
    sqrt_true = torch.sqrt(torch.clamp(y_true, min=1e-8))
    sqrt_pred = torch.sqrt(torch.clamp(y_pred, min=1e-8))
    
    # Components of the loss
    diff = sqrt_pred - sqrt_true
    term1 = 10 * torch.square(torch.square(diff))  # 10 * (sqrt_diff)^4
    term2 = torch.square(diff)  # (sqrt_diff)^2
    
    # For BCE, we need to handle negative values in y_true
    # Convert y_true to [0, 1] range for BCE: (y_true + 1) / 2
    y_true_bce = (y_true + 1.0) / 2.0
    y_pred_bce = (y_pred + 1.0) / 2.0
    term3 = 0.01 * F.binary_cross_entropy(y_pred_bce, y_true_bce, reduction='none')  # 0.01 * BCE
    
    return torch.mean(mask * (term1 + term2 + term3))


def custom_vad_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Custom VAD crossentropy equivalent to Keras my_crossentropy"""
    weight = 2 * torch.abs(y_true - 0.5)
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    # Ensure weight and bce have compatible shapes
    if weight.dim() > bce.dim():
        weight = weight.squeeze(-1)  # Remove last dimension if it's 1
    return torch.mean(weight * bce)


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                device: torch.device,
                denoise_weight: float = 10.0,
                vad_weight: float = 0.5) -> Tuple[float, float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_denoise_loss = 0.0
    total_vad_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y_denoise, y_vad) in enumerate(dataloader):
        x = x.to(device)
        y_denoise = y_denoise.to(device)
        y_vad = y_vad.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_denoise, pred_vad = model(x)
        
        # Compute losses
        denoise_loss = custom_denoise_loss(y_denoise, pred_denoise)
        vad_loss = custom_vad_crossentropy(y_vad, pred_vad)
        
        # Combined loss with weights
        total_batch_loss = denoise_weight * denoise_loss + vad_weight * vad_loss
        
        # Backward pass
        total_batch_loss.backward()
        
        # Apply weight constraints before optimizer step
        model._apply_constraints()
        
        optimizer.step()
        
        # Apply weight constraints after optimizer step  
        model._apply_constraints()
        
        total_loss += total_batch_loss.item()
        total_denoise_loss += denoise_loss.item()
        total_vad_loss += vad_loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}: '
                  f'Total Loss: {total_batch_loss.item():.6f}, '
                  f'Denoise: {denoise_loss.item():.6f}, '
                  f'VAD: {vad_loss.item():.6f}')
    
    return (total_loss / num_batches, 
            total_denoise_loss / num_batches, 
            total_vad_loss / num_batches)


def validate_epoch(model: nn.Module, 
                   dataloader: DataLoader, 
                   device: torch.device,
                   denoise_weight: float = 10.0,
                   vad_weight: float = 0.5) -> Tuple[float, float, float]:
    """Validate model for one epoch"""
    model.eval()
    total_loss = 0.0
    total_denoise_loss = 0.0
    total_vad_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y_denoise, y_vad in dataloader:
            x = x.to(device)
            y_denoise = y_denoise.to(device)
            y_vad = y_vad.to(device)
            
            # Forward pass
            pred_denoise, pred_vad = model(x)
            
            # Compute losses
            denoise_loss = custom_denoise_loss(y_denoise, pred_denoise)
            vad_loss = custom_vad_crossentropy(y_vad, pred_vad)
            
            # Combined loss with weights
            total_batch_loss = denoise_weight * denoise_loss + vad_weight * vad_loss
            
            total_loss += total_batch_loss.item()
            total_denoise_loss += denoise_loss.item()
            total_vad_loss += vad_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return (0.0, 0.0, 0.0)
    
    return (total_loss / num_batches, 
            total_denoise_loss / num_batches, 
            total_vad_loss / num_batches)


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNNoise Training')
    parser.add_argument('--data-file', type=str, default='training.h5',
                        help='Path to training data file (HDF5 or binary format)')
    parser.add_argument('--data-format', type=str, choices=['hdf5', 'binary'], default='hdf5',
                        help='Data format: hdf5 (original) or binary (denoise.c output)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--window-size', type=int, default=2000,
                        help='Sequence window size')
    parser.add_argument('--output-model', type=str, default='rnnoise_pytorch.pth',
                        help='Output model file name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cpu, cuda, mps')
    parser.add_argument('--l2-reg', type=float, default=1e-6,
                        help='L2 regularization factor')
    parser.add_argument('--constraint-clip', type=float, default=0.499,
                        help='Weight constraint clipping value')
    parser.add_argument('--denoise-weight', type=float, default=10.0,
                        help='Loss weight for denoise output')
    parser.add_argument('--vad-weight', type=float, default=0.5,
                        help='Loss weight for VAD output')
    parser.add_argument('--help-binary', action='store_true',
                        help='Show help for creating binary training data')
    
    args = parser.parse_args()
    
    # Show binary data creation help if requested
    if args.help_binary:
        create_binary_training_data()
        return
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Load dataset
    full_dataset = RNNoiseDataset(args.data_file, args.window_size, args.data_format)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(args.validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f'Training samples: {train_size}, Validation samples: {val_size}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Create model
    model = RNNoiseModel(l2_reg=args.l2_reg, constraint_clip=args.constraint_clip)
    model = model.to(device)
    
    # Create optimizer (Adam equivalent to Keras)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    
    print('Starting training...')
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Training
        train_loss, train_denoise, train_vad = train_epoch(
            model, train_loader, optimizer, device, 
            args.denoise_weight, args.vad_weight
        )
        
        # Validation
        val_loss, val_denoise, val_vad = validate_epoch(
            model, val_loader, device,
            args.denoise_weight, args.vad_weight
        )
        
        print(f'Train Loss: {train_loss:.6f} (Denoise: {train_denoise:.6f}, VAD: {train_vad:.6f})')
        print(f'Val Loss: {val_loss:.6f} (Denoise: {val_denoise:.6f}, VAD: {val_vad:.6f})')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, args.output_model)
            print(f'New best model saved: {args.output_model}')
    
    print(f'\nTraining completed. Best validation loss: {best_val_loss:.6f}')
    print(f'Model saved as: {args.output_model}')


def create_binary_training_data():
    """
    Example function showing how to create binary training data using denoise.c
    This is for reference only - you need to compile and run denoise.c with TRAINING=1
    """
    print("To create binary training data:")
    print("1. Compile denoise.c with TRAINING=1:")
    print("   cd src && gcc -DTRAINING=1 -O3 denoise.c ... -o denoise_training")
    print("")
    print("2. Run with clean speech and noise files:")
    print("   ./denoise_training speech.raw noise.raw 100000 > training_data.bin")
    print("")
    print("3. Train with PyTorch using binary format:")
    print("   python rnn_train_pytorch.py --data-format binary --data-file training_data.bin")
    print("")
    print("Note: speech.raw and noise.raw should be 16-bit PCM files")


if __name__ == '__main__':
    main()