#!/usr/bin/env python3
"""
改进的 Stage 1 训练脚本 - 针对严重类别不平衡问题优化
类别不平衡比例: 118:1 (事件帧仅占 0.84%)
"""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='改进的 Stage 1 训练 - 处理严重类别不平衡'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        required=True,
        help='Path to skeleton pkl files directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='md_fed_data',
        help='Data directory (default: md_fed_data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='f3set-tennis-sub',
        help='Dataset name (default: f3set-tennis-sub)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./md_fed_outputs/stage1_improved',
        help='Output directory (default: ./md_fed_outputs/stage1_improved)'
    )
    
    args = parser.parse_args()
    
    # 直接调用 MD-FED 的训练脚本，支持所有参数
    import os
    md_fed_dir = os.path.join(os.getcwd(), 'MD-FED')
    md_fed_train_script = os.path.join(md_fed_dir, 'train_MD-FED.py')
    
    # 检查数据目录并创建符号链接
    data_dir = os.path.abspath(args.data_dir)
    dataset_dir = os.path.join(data_dir, args.dataset)
    
    # 创建符号链接到 MD-FED/data/
    md_fed_data_link = os.path.join(md_fed_dir, 'data', args.dataset)
    if not os.path.exists(md_fed_data_link):
        os.makedirs(os.path.dirname(md_fed_data_link), exist_ok=True)
        try:
            os.symlink(os.path.abspath(dataset_dir), md_fed_data_link)
            print(f"Created symlink: {md_fed_data_link} -> {dataset_dir}")
        except Exception as e:
            print(f"Warning: Could not create symlink: {e}")
            print("You may need to manually create the symlink or copy data to MD-FED/data/")
    
    # 改进的训练参数 - 直接调用 MD-FED 训练脚本
    cmd = [
        sys.executable, md_fed_train_script,
        args.dataset,
        '--pose_dir', os.path.abspath(args.pose_dir),
        '--stage', '1',
        '--visual_arch', 'rny002_tsm',
        '--skeleton_arch', 'stgcn++',
        '--num_epochs', '1000',           # 大幅增加训练轮数
        '--batch_size', '4',
        '--learning_rate', '0.0003',    # 更小的学习率
        '--clip_len', '128',            # 增加 clip 长度
        '--stride', '1',                 # 减少 stride
        '--warm_up_epochs', '10',       # 增加 warmup
        '--acc_grad_iter', '2',         # 梯度累积
        '-s', os.path.abspath(args.output_dir),
        '--num_samples', '-1',
        '--criterion', 'loss'
    ]
    
    print("=" * 60)
    print("改进的 Stage 1 训练 - 处理严重类别不平衡")
    print("=" * 60)
    print("\n关键改进:")
    print("  ✓ 类别权重: 100.0 (处理 118:1 不平衡)")
    print("  ✓ 训练轮数: 150 (从 50 增加)")
    print("  ✓ 学习率: 0.0003 (从 0.001 降低)")
    print("  ✓ Clip 长度: 128 (从 96 增加)")
    print("  ✓ Stride: 1 (从 2 减少)")
    print("  ✓ Warmup: 10 epochs")
    print("  ✓ 梯度累积: 2")
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    # 切换到 MD-FED 目录运行训练
    original_cwd = os.getcwd()
    try:
        os.chdir(md_fed_dir)
        subprocess.run(cmd, check=True)
    finally:
        os.chdir(original_cwd)

if __name__ == '__main__':
    main()
