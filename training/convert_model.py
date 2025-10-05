#!/usr/bin/env python3
"""
简化版的PyTorch模型转换脚本
一键将weights.pth转换为C格式
"""

import os
import sys
import subprocess

def main():
    # 检查输入文件
    model_file = "weights.pth"
    if not os.path.exists(model_file):
        print(f"错误: 找不到模型文件 {model_file}")
        print("请先运行训练脚本生成模型文件")
        sys.exit(1)
    
    # 输出文件名
    c_file = "rnnoise_weights.c"
    bin_file = "rnnoise_weights.bin"
    
    print("开始转换PyTorch模型到C格式...")
    print(f"输入文件: {model_file}")
    print(f"输出文件: {c_file}, {bin_file}")
    
    try:
        # 运行转换脚本
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "dump_pytorch_model_fixed.py")
        
        cmd = [
            "python3", script_path,
            model_file,
            c_file, 
            bin_file,
            "--model-name", "pytorch"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 转换成功!")
            print(f"生成文件:")
            print(f"  - {c_file}")
            print(f"  - {bin_file}")
            
            # 显示文件大小
            if os.path.exists(c_file):
                size = os.path.getsize(c_file)
                print(f"  C文件大小: {size:,} 字节")
            
            if os.path.exists(bin_file):
                size = os.path.getsize(bin_file)
                print(f"  二进制文件大小: {size:,} 字节")
            
            print("\n下一步:")
            print("1. 将生成的文件复制到src/目录")
            print("2. 修改rnn_data.c包含新模型")
            print("3. 重新编译RNNoise项目")
            
        else:
            print("❌ 转换失败!")
            print("错误信息:")
            print(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
