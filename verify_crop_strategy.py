#!/usr/bin/env python3
"""
验证训练和评估时使用的裁剪策略是否正确配置

预期结果:
- 训练集: RandomCrop(size=(224, 224))
- 验证集: CenterCrop(size=(224, 224))
"""

import os
import sys
import argparse
from pathlib import Path

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

from dataset.input_process import ActionSeqDataset
from util.dataset import load_classes


def verify_crop_strategy(data_dir, frame_dir, crop_dim=224, clip_len=96):
    """
    验证裁剪策略配置
    
    Args:
        data_dir: 包含 train.json, val.json 和 elements.txt 的目录
        frame_dir: 帧目录
        crop_dim: 裁剪尺寸
        clip_len: 片段长度
    """
    print(f"{'='*80}")
    print("裁剪策略验证工具")
    print(f"{'='*80}\n")
    
    # Load classes
    elements_file = os.path.join(data_dir, 'elements.txt')
    if not os.path.exists(elements_file):
        print(f"❌ 错误: 找不到 {elements_file}")
        return False
    
    classes = load_classes(elements_file)
    print(f"✓ 加载了 {len(classes)} 个类别")
    
    # Check if data files exist
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    if not os.path.exists(train_json):
        print(f"❌ 错误: 找不到 {train_json}")
        return False
    if not os.path.exists(val_json):
        print(f"❌ 错误: 找不到 {val_json}")
        return False
    
    print(f"✓ 找到训练和验证数据文件\n")
    
    # Create training dataset
    print(f"{'─'*80}")
    print("1. 训练集配置")
    print(f"{'─'*80}")
    
    try:
        train_data = ActionSeqDataset(
            classes, train_json,
            frame_dir, clip_len, 100,
            is_eval=False,  # 训练时：随机裁剪
            dilate_len=0, stage=3,
            num_samples=-1, flow_dir=None, pose_dir=None,
            crop_dim=crop_dim, stride=2
        )
        
        train_crop = train_data._frame_reader._crop_transform
        print(f"参数: is_eval=False, crop_dim={crop_dim}")
        print(f"裁剪策略: {train_crop}")
        print(f"策略类型: {type(train_crop).__name__}")
        
        # Check if it's RandomCrop
        if 'RandomCrop' in str(type(train_crop).__name__):
            print("✅ 正确：使用随机裁剪 (RandomCrop)")
            train_correct = True
        else:
            print("❌ 错误：应该使用随机裁剪 (RandomCrop)")
            train_correct = False
            
    except Exception as e:
        print(f"❌ 创建训练集失败: {e}")
        train_correct = False
    
    # Create validation dataset
    print(f"\n{'─'*80}")
    print("2. 验证集配置")
    print(f"{'─'*80}")
    
    try:
        val_data = ActionSeqDataset(
            classes, val_json,
            frame_dir, clip_len, 25,
            is_eval=True,  # 验证时：中心裁剪
            dilate_len=0, stage=3,
            num_samples=-1, flow_dir=None, pose_dir=None,
            crop_dim=crop_dim, stride=2
        )
        
        val_crop = val_data._frame_reader._crop_transform
        print(f"参数: is_eval=True, crop_dim={crop_dim}")
        print(f"裁剪策略: {val_crop}")
        print(f"策略类型: {type(val_crop).__name__}")
        
        # Check if it's CenterCrop
        if 'CenterCrop' in str(type(val_crop).__name__):
            print("✅ 正确：使用中心裁剪 (CenterCrop)")
            val_correct = True
        else:
            print("❌ 错误：应该使用中心裁剪 (CenterCrop)")
            val_correct = False
            
    except Exception as e:
        print(f"❌ 创建验证集失败: {e}")
        val_correct = False
    
    # Summary
    print(f"\n{'='*80}")
    print("验证结果总结")
    print(f"{'='*80}\n")
    
    print(f"{'配置':<15} {'预期策略':<20} {'实际策略':<20} {'状态':<10}")
    print(f"{'─'*80}")
    
    train_type = type(train_crop).__name__ if train_correct or 'train_crop' in locals() else "N/A"
    val_type = type(val_crop).__name__ if val_correct or 'val_crop' in locals() else "N/A"
    
    print(f"{'训练集':<15} {'RandomCrop':<20} {train_type:<20} {'✅' if train_correct else '❌':<10}")
    print(f"{'验证集':<15} {'CenterCrop':<20} {val_type:<20} {'✅' if val_correct else '❌':<10}")
    
    # Overall result
    all_correct = train_correct and val_correct
    
    print(f"\n{'='*80}")
    if all_correct:
        print("✅ 所有配置正确！可以进行公平对比实验。")
    else:
        print("❌ 配置有误！请检查代码中的 is_eval 参数设置。")
    print(f"{'='*80}\n")
    
    # Additional info
    if all_correct:
        print("📊 裁剪行为说明:\n")
        print("训练时 (RandomCrop):")
        print("  - 从 398×224 帧中随机裁剪 224×224")
        print("  - 每个 epoch 看到不同的区域")
        print("  - 提供数据增强，减少过拟合\n")
        print("评估时 (CenterCrop):")
        print("  - 从 398×224 帧中固定裁剪中心 224×224")
        print("  - 确定性推理，结果可复现")
        print("  - 与你的现有方法保持一致\n")
    
    return all_correct


def main():
    parser = argparse.ArgumentParser(
        description='验证训练和评估时的裁剪策略配置'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='vtn_data',
        help='包含 train.json, val.json 和 elements.txt 的目录'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='帧目录路径'
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=224,
        help='裁剪尺寸 (默认: 224)'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='片段长度 (默认: 96)'
    )
    
    args = parser.parse_args()
    
    # Verify paths
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误: 数据目录不存在: {args.data_dir}")
        print(f"\n💡 提示: 先运行训练脚本生成数据:")
        print(f"   python train_vtn_comparison.py --manual_annotations manual_annotations.json ...")
        sys.exit(1)
    
    if not os.path.exists(args.frame_dir):
        print(f"❌ 错误: 帧目录不存在: {args.frame_dir}")
        sys.exit(1)
    
    # Run verification
    success = verify_crop_strategy(
        args.data_dir,
        args.frame_dir,
        args.crop_dim,
        args.clip_len
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
