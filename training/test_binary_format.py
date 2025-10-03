#!/usr/bin/env python3
"""
Test script to verify binary data format reading
"""

import numpy as np
import argparse
import os
import sys

def test_binary_format(binary_file: str, num_frames_to_show: int = 10):
    """
    Test reading binary format data from denoise.c output
    
    Args:
        binary_file: Path to binary file created by denoise.c
        num_frames_to_show: Number of sample frames to display
    """
    if not os.path.exists(binary_file):
        print(f"Error: File {binary_file} does not exist")
        return False
    
    print(f"Testing binary file: {binary_file}")
    
    # Each frame contains: 42 (features) + 22 (denoise_targets) + 22 (noise_targets) + 1 (vad) = 87 floats
    frame_size = 87
    
    # Read binary data
    with open(binary_file, 'rb') as f:
        data_bytes = f.read()
    
    file_size = len(data_bytes)
    total_floats = file_size // 4  # 4 bytes per float
    total_frames = total_floats // frame_size
    
    print(f"File size: {file_size} bytes")
    print(f"Total floats: {total_floats}")
    print(f"Expected frame size: {frame_size} floats")
    print(f"Total frames: {total_frames}")
    
    if total_floats % frame_size != 0:
        print(f"Warning: File size is not divisible by frame size!")
        print(f"Remainder: {total_floats % frame_size} floats")
        return False
    
    if total_frames == 0:
        print("Error: No complete frames found")
        return False
    
    # Load as float32 array
    all_data = np.frombuffer(data_bytes, dtype=np.float32)
    all_data = all_data.reshape(total_frames, frame_size)
    
    print(f"Successfully loaded {total_frames} frames")
    print(f"Data shape: {all_data.shape}")
    
    # Extract components
    features = all_data[:, :42]      # Features (42 dims)
    denoise_targets = all_data[:, 42:64]   # Denoise targets (22 dims)
    noise_targets = all_data[:, 64:86]     # Noise targets (22 dims)  
    vad_targets = all_data[:, 86:87]       # VAD targets (1 dim)
    
    print(f"\nComponent shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Denoise targets: {denoise_targets.shape}")
    print(f"  Noise targets: {noise_targets.shape}")
    print(f"  VAD targets: {vad_targets.shape}")
    
    print(f"\nData ranges:")
    print(f"  Features: [{np.min(features):.3f}, {np.max(features):.3f}]")
    print(f"  Denoise targets: [{np.min(denoise_targets):.3f}, {np.max(denoise_targets):.3f}]")
    print(f"  Noise targets: [{np.min(noise_targets):.3f}, {np.max(noise_targets):.3f}]")
    print(f"  VAD targets: [{np.min(vad_targets):.3f}, {np.max(vad_targets):.3f}]")
    
    # Show sample frames
    num_to_show = min(num_frames_to_show, total_frames)
    print(f"\nSample frames (first {num_to_show}):")
    
    for i in range(num_to_show):
        print(f"\nFrame {i}:")
        print(f"  Features[:5]: {features[i, :5]}")
        print(f"  Denoise targets[:5]: {denoise_targets[i, :5]}")
        print(f"  VAD target: {vad_targets[i, 0]}")
    
    return True

def create_test_binary_file(output_file: str, num_frames: int = 1000):
    """
    Create a test binary file with random data for testing
    
    Args:
        output_file: Output file path
        num_frames: Number of frames to generate
    """
    print(f"Creating test binary file: {output_file}")
    
    frame_size = 87
    
    # Generate random test data
    np.random.seed(42)  # For reproducible results
    
    # Features: typically in range [-10, 10]
    features = np.random.normal(0, 3, (num_frames, 42)).astype(np.float32)
    
    # Denoise targets: typically in range [0, 1] (gains)
    denoise_targets = np.random.uniform(0, 1, (num_frames, 22)).astype(np.float32)
    
    # Noise targets: typically log energy values in range [-5, 5]
    noise_targets = np.random.normal(0, 2, (num_frames, 22)).astype(np.float32)
    
    # VAD targets: binary values [0, 1]
    vad_targets = np.random.choice([0.0, 1.0], (num_frames, 1)).astype(np.float32)
    
    # Combine all data
    all_data = np.concatenate([features, denoise_targets, noise_targets, vad_targets], axis=1)
    
    print(f"Generated data shape: {all_data.shape}")
    
    # Write to binary file
    with open(output_file, 'wb') as f:
        all_data.tobytes()
        f.write(all_data.tobytes())
    
    print(f"Test file created: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test binary format reading for RNNoise')
    parser.add_argument('--binary-file', type=str, 
                        help='Path to binary file to test')
    parser.add_argument('--create-test', type=str,
                        help='Create a test binary file at specified path')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='Number of frames for test file creation')
    parser.add_argument('--show-frames', type=int, default=5,
                        help='Number of sample frames to display')
    
    args = parser.parse_args()
    
    if args.create_test:
        if create_test_binary_file(args.create_test, args.num_frames):
            print(f"\nTest file created successfully!")
            if input("Test the created file? (y/n): ").lower() == 'y':
                test_binary_format(args.create_test, args.show_frames)
    elif args.binary_file:
        if test_binary_format(args.binary_file, args.show_frames):
            print(f"\nBinary file format is valid!")
        else:
            print(f"\nBinary file format test failed!")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Create a test file")
        print("  python test_binary_format.py --create-test test_data.bin --num-frames 5000")
        print("  # Test an existing file")
        print("  python test_binary_format.py --binary-file training_data.bin")

if __name__ == '__main__':
    main()