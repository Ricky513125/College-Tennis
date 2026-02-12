#!/usr/bin/env python3
"""
检查视频帧的尺寸并推荐最佳的矩形输入配置
"""

import os
import sys
from PIL import Image
from collections import Counter
import argparse


def analyze_frame_sizes(frame_dir, num_samples=100):
    """
    分析帧尺寸并推荐配置
    
    Args:
        frame_dir: 帧目录
        num_samples: 采样数量
    """
    print(f"{'='*80}")
    print(f"帧尺寸分析工具")
    print(f"{'='*80}\n")
    
    print(f"正在扫描目录: {frame_dir}")
    print(f"采样数量: {num_samples}\n")
    
    sizes = []
    sampled_files = []
    
    # 扫描帧
    for root, dirs, files in os.walk(frame_dir):
        for file in sorted(files):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    sizes.append((img.width, img.height))
                    sampled_files.append(img_path)
                    
                    if len(sizes) <= 5:
                        print(f"✓ {img_path}: {img.width}×{img.height}")
                    
                    if len(sizes) >= num_samples:
                        break
                except Exception as e:
                    print(f"⚠️  无法读取 {img_path}: {e}")
        
        if len(sizes) >= num_samples:
            break
    
    if len(sizes) == 0:
        print(f"\n❌ 错误: 在 {frame_dir} 中未找到任何图像文件")
        return None, None
    
    print(f"\n已采样 {len(sizes)} 张图像")
    
    # 统计尺寸
    size_counts = Counter(sizes)
    print(f"\n{'='*80}")
    print("帧尺寸统计:")
    print(f"{'='*80}")
    
    for size, count in size_counts.most_common():
        percentage = count / len(sizes) * 100
        print(f"  {size[0]}×{size[1]}: {count} 张 ({percentage:.1f}%)")
    
    # 获取最常见的尺寸
    most_common_size = size_counts.most_common(1)[0][0]
    width, height = most_common_size
    
    print(f"\n{'='*80}")
    print(f"检测到的主要尺寸: {width}×{height}")
    print(f"{'='*80}\n")
    
    # 计算最佳矩形尺寸（必须能被16整除）
    def round_down_to_multiple(n, base=16):
        """向下取整到base的倍数"""
        return (n // base) * base
    
    def round_up_to_multiple(n, base=16):
        """向上取整到base的倍数"""
        return ((n + base - 1) // base) * base
    
    # 方案1: 向下取整（resize到更小）
    best_width_down = round_down_to_multiple(width)
    best_height_down = round_down_to_multiple(height)
    
    # 方案2: 向上取整（需要padding）
    best_width_up = round_up_to_multiple(width)
    best_height_up = round_up_to_multiple(height)
    
    print("📊 推荐配置\n")
    
    # 推荐方案1（向下取整 - 推荐）
    print(f"方案1: 向下 Resize (推荐 ⭐)")
    print(f"{'─'*80}")
    print(f"配置参数:")
    print(f"  --img_width {best_width_down} \\")
    print(f"  --img_height {best_height_down}")
    
    width_loss = width - best_width_down
    height_loss = height - best_height_down
    width_loss_pct = (width_loss / width * 100) if width > 0 else 0
    height_loss_pct = (height_loss / height * 100) if height > 0 else 0
    
    print(f"\n信息损失:")
    print(f"  宽度: {width} → {best_width_down} (损失 {width_loss}px, {width_loss_pct:.1f}%)")
    print(f"  高度: {height} → {best_height_down} (损失 {height_loss}px, {height_loss_pct:.1f}%)")
    
    num_patches = (best_width_down // 16) * (best_height_down // 16)
    print(f"\nVision Transformer:")
    print(f"  Patches: {best_width_down // 16}×{best_height_down // 16} = {num_patches} 个")
    print(f"  操作: Resize (无需padding)")
    
    # 推荐方案2（向上取整）
    if best_width_up != width or best_height_up != height:
        print(f"\n\n方案2: 向上 Padding")
        print(f"{'─'*80}")
        print(f"配置参数:")
        print(f"  --img_width {best_width_up} \\")
        print(f"  --img_height {best_height_up}")
        
        width_pad = best_width_up - width
        height_pad = best_height_up - height
        
        print(f"\nPadding需求:")
        print(f"  宽度: {width} → {best_width_up} (padding {width_pad}px)")
        print(f"  高度: {height} → {best_height_up} (padding {height_pad}px)")
        
        num_patches_up = (best_width_up // 16) * (best_height_up // 16)
        print(f"\nVision Transformer:")
        print(f"  Patches: {best_width_up // 16}×{best_height_up // 16} = {num_patches_up} 个")
        print(f"  操作: Padding + Resize")
    
    # 正方形对比
    print(f"\n\n对比: 正方形输入 (传统方法)")
    print(f"{'─'*80}")
    
    # 使用高度作为正方形尺寸（通常高度较小）
    square_size = round_down_to_multiple(min(width, height))
    print(f"正方形尺寸: {square_size}×{square_size}")
    print(f"配置参数: --crop_dim {square_size}")
    
    square_width_loss = width - square_size
    square_height_loss = height - square_size
    square_width_loss_pct = (square_width_loss / width * 100) if width > 0 else 0
    square_height_loss_pct = (square_height_loss / height * 100) if height > 0 else 0
    
    print(f"\n信息损失:")
    print(f"  宽度: {width} → {square_size} (损失 {square_width_loss}px, {square_width_loss_pct:.1f}%)")
    print(f"  高度: {height} → {square_size} (损失 {square_height_loss}px, {square_height_loss_pct:.1f}%)")
    
    square_patches = (square_size // 16) ** 2
    print(f"\nVision Transformer:")
    print(f"  Patches: {square_size // 16}×{square_size // 16} = {square_patches} 个")
    
    # 总结对比
    print(f"\n\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}\n")
    
    print(f"{'配置':<20} {'尺寸':<15} {'损失':<15} {'Patches':<10} {'推荐':<10}")
    print(f"{'─'*80}")
    
    rect_loss = f"{width_loss_pct:.1f}%×{height_loss_pct:.1f}%"
    print(f"{'矩形(向下)':<20} {f'{best_width_down}×{best_height_down}':<15} {rect_loss:<15} {num_patches:<10} {'⭐⭐⭐⭐⭐':<10}")
    
    if best_width_up != width or best_height_up != height:
        rect_up_loss = f"0%(pad)"
        print(f"{'矩形(向上)':<20} {f'{best_width_up}×{best_height_up}':<15} {rect_up_loss:<15} {num_patches_up:<10} {'⭐⭐⭐⭐':<10}")
    
    square_loss = f"{square_width_loss_pct:.1f}%×{square_height_loss_pct:.1f}%"
    print(f"{'正方形':<20} {f'{square_size}×{square_size}':<15} {square_loss:<15} {square_patches:<10} {'⭐⭐⭐':<10}")
    
    # 完整命令示例
    print(f"\n\n{'='*80}")
    print("完整训练命令示例")
    print(f"{'='*80}\n")
    
    print("# 推荐: 使用矩形输入保留最多信息")
    print(f"""python train_vtn_comparison.py \\
    --manual_annotations manual_annotations.json \\
    --frame_dir {frame_dir} \\
    --save_dir ./vtn_outputs/rect_{best_width_down}x{best_height_down} \\
    --img_width {best_width_down} \\
    --img_height {best_height_down} \\
    --patch_size 16 \\
    --vtn_spatial_size small \\
    --num_epochs 50
""")
    
    print("\n# 备选: 使用正方形输入（更传统）")
    print(f"""python train_vtn_comparison.py \\
    --manual_annotations manual_annotations.json \\
    --frame_dir {frame_dir} \\
    --save_dir ./vtn_outputs/square_{square_size} \\
    --crop_dim {square_size} \\
    --vtn_spatial_size small \\
    --num_epochs 50
""")
    
    print(f"{'='*80}\n")
    
    return best_width_down, best_height_down


def main():
    parser = argparse.ArgumentParser(
        description='分析视频帧尺寸并推荐最佳VTN配置'
    )
    parser.add_argument(
        'frame_dir',
        type=str,
        help='帧目录路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='采样数量（默认：100）'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frame_dir):
        print(f"错误: 目录不存在: {args.frame_dir}")
        sys.exit(1)
    
    analyze_frame_sizes(args.frame_dir, args.num_samples)


if __name__ == '__main__':
    main()
