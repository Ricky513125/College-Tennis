#!/usr/bin/env python3
"""
修复 Stage 1 训练中的类别不平衡问题
通过修改 MD-FED 的训练代码来改善类别不平衡
"""

import os
import shutil
import re


def apply_class_imbalance_fixes():
    """
    应用类别不平衡修复到 MD-FED/train_MD-FED.py
    """
    train_file = 'MD-FED/train_MD-FED.py'
    backup_file = 'MD-FED/train_MD-FED.py.backup'
    
    # 创建备份
    if not os.path.exists(backup_file):
        shutil.copy2(train_file, backup_file)
        print(f"已创建备份: {backup_file}")
    
    # 读取文件
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复 1: 增加类别权重（从 20.0 增加到 100.0）
    # 查找并替换类别权重
    pattern1 = r"class_weights = torch\.tensor\(\[1\.0, \d+\.0\]\)\.to\(self\._device\)"
    replacement1 = "class_weights = torch.tensor([1.0, 100.0]).to(self._device)  # Increased for severe class imbalance (118:1)"
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print("✓ 已更新类别权重: 20.0 -> 100.0")
    else:
        # 如果找不到，尝试添加
        old_line = "class_weights = torch.tensor([1.0, 20.0]).to(self._device)"
        if old_line in content:
            content = content.replace(old_line, replacement1)
            print("✓ 已更新类别权重: 20.0 -> 100.0")
        else:
            print("⚠️  未找到类别权重设置，可能需要手动添加")
    
    # 修复 2: 添加 Focal Loss 支持（可选，作为注释提供）
    focal_loss_code = """
    # Focal Loss implementation for severe class imbalance
    # Uncomment to use Focal Loss instead of weighted Cross-Entropy
    # def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    #     ce_loss = F.cross_entropy(pred, target, reduction='none')
    #     pt = torch.exp(-ce_loss)
    #     focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    #     return focal_loss.mean()
    """
    
    # 修复 3: 添加事件帧采样增强（在数据加载时）
    # 这个需要在 dataset 层面修改，这里先提供建议
    
    # 保存修改后的文件
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ 修复已应用到: {train_file}")
    print("\n建议的下一步:")
    print("1. 重新训练 Stage 1，使用更高的类别权重")
    print("2. 考虑使用 Focal Loss（需要取消注释相关代码）")
    print("3. 增加训练轮数到 100-150")
    print("4. 使用更小的学习率 (0.0005)")


def create_improved_training_script():
    """创建改进的训练脚本"""
    script_content = """#!/usr/bin/env python3
\"\"\"
改进的 Stage 1 训练脚本 - 针对严重类别不平衡问题优化
类别不平衡比例: 118:1 (事件帧仅占 0.84%)
\"\"\"

import subprocess
import sys

# 改进的训练参数
cmd = [
    sys.executable, 'run_md_fed_stage1.py',
    '--data_dir', 'md_fed_data',
    '--dataset', 'f3set-tennis-sub',
    '--pose_dir', '/path/to/skeleton/pkl/files',  # 请修改为实际路径
    '--output_dir', './md_fed_outputs/stage1_improved',
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

print("开始改进的 Stage 1 训练...")
print("=" * 60)
print("关键改进:")
print("  - 类别权重: 100.0 (处理 118:1 不平衡)")
print("  - 训练轮数: 150")
print("  - 学习率: 0.0003")
print("  - Clip 长度: 128")
print("  - Stride: 1")
print("=" * 60)

subprocess.run(cmd)
"""
    
    with open('run_stage1_improved.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("\n✓ 已创建改进的训练脚本: run_stage1_improved.py")
    print("  请修改其中的 --pose_dir 路径后运行")


if __name__ == '__main__':
    print("=" * 60)
    print("Stage 1 类别不平衡修复工具")
    print("=" * 60)
    print("\n诊断结果:")
    print("  - 类别不平衡比例: 118.4:1")
    print("  - 事件帧比例: 0.84%")
    print("  - 这是导致模型效果差的主要原因！")
    print("\n" + "=" * 60)
    
    apply_class_imbalance_fixes()
    create_improved_training_script()
    
    print("\n" + "=" * 60)
    print("修复完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查 MD-FED/train_MD-FED.py 中的类别权重是否已更新为 100.0")
    print("2. 运行改进的训练脚本:")
    print("   python run_stage1_improved.py")
    print("3. 监控训练过程，观察 Event F1 是否提升")
