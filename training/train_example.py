#!/usr/bin/env python3
"""
Example script showing how to use the PyTorch RNNoise training script
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"\n{description}")
    print(f"Running: {cmd}")
    print("-" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True


def main():
    """Main training example"""
    
    print("RNNoise PyTorch Training Example")
    print("=" * 40)
    
    # Check if training data exists
    if not os.path.exists('training.h5'):
        print("\nERROR: training.h5 not found!")
        print("Please prepare your training data in HDF5 format.")
        print("The data should have shape (N, 87) where:")
        print("  - Columns 0-41: Input features (42 dimensions)")
        print("  - Columns 42-63: Denoise targets (22 dimensions)")  
        print("  - Columns 64-85: Noise targets (22 dimensions, optional)")
        print("  - Column 86: VAD targets (1 dimension)")
        sys.exit(1)
    
    # Basic training command
    basic_cmd = ("python3 rnn_train_pytorch.py "
                "--data-file training.h5 "
                "--batch-size 32 "
                "--epochs 10 "  # Reduced for example
                "--lr 0.001 "
                "--output-model rnnoise_test.pth")
    
    if not run_command(basic_cmd, "1. Basic training (10 epochs for testing)"):
        print("Training failed!")
        sys.exit(1)
    
    # Check if model was created
    if os.path.exists('rnnoise_test.pth'):
        print("\n✓ Training completed successfully!")
        print("Model saved as: rnnoise_test.pth")
        
        # Convert to C format
        convert_cmd = ("python3 dump_pytorch_model.py "
                      "rnnoise_test.pth "
                      "rnnoise_pytorch_weights.c "
                      "rnnoise_pytorch_weights.bin "
                      "--model-name test")
        
        if run_command(convert_cmd, "2. Converting model to C format"):
            print("\n✓ Model conversion completed!")
            print("C weights file: rnnoise_pytorch_weights.c")
            print("Binary weights file: rnnoise_pytorch_weights.bin")
            
            # Show usage instructions
            print("\n" + "=" * 50)
            print("USAGE INSTRUCTIONS")
            print("=" * 50)
            print("\nTo use the trained model in your C application:")
            print("1. Include the generated C file in your build")
            print("2. Use the model with: rnnoise_model_test")
            print("\nFor full training (120 epochs), run:")
            print("python3 rnn_train_pytorch.py --epochs 120 --output-model rnnoise_full.pth")
            
        else:
            print("Model conversion failed!")
    else:
        print("Training failed - no model file created")


if __name__ == '__main__':
    main()