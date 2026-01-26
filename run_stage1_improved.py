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
    
    # 改进的训练参数
    cmd = [
        sys.executable, 'run_md_fed_stage1.py',
        '--data_dir', args.data_dir,
        '--dataset', args.dataset,
        '--pose_dir', args.pose_dir,
        '--output_dir', args.output_dir,
        '--num_epochs', '150',           # 大幅增加训练轮数
        '--batch_size', '4',
        '--learning_rate', '0.0003',    # 更小的学习率
        '--clip_len', '128',            # 增加 clip 长度
        '--stride', '1',                 # 减少 stride
        '--warm_up_epochs', '10',       # 增加 warmup
        '--acc_grad_iter', '2',         # 梯度累积
        '--visual_arch', 'rny002_tsm',
        '--skeleton_arch', 'stgcn++'
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
    
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
