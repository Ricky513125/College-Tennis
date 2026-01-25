#!/usr/bin/env python3
"""
诊断为什么模型没有预测任何事件
"""

import sys
import json
import numpy as np

def diagnose_no_predictions(loss_file, output_dir):
    """诊断为什么模型不预测事件"""
    
    print("="*60)
    print("诊断：为什么模型没有预测任何事件")
    print("="*60)
    
    # 检查损失文件
    if loss_file and os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            losses = json.load(f)
        
        train_losses = [l['train'] for l in losses]
        val_losses = [l['val'] for l in losses]
        
        print(f"\n训练损失趋势:")
        print(f"  初始: {train_losses[0]:.5f}")
        print(f"  最终: {train_losses[-1]:.5f}")
        print(f"  变化: {train_losses[0] - train_losses[-1]:+.5f}")
        
        if train_losses[-1] >= train_losses[0]:
            print("  ⚠️  损失没有下降 - 模型可能没有学习")
        else:
            print("  ✓ 损失在下降 - 模型在学习，但可能学错了")
    
    print("\n" + "="*60)
    print("可能的原因:")
    print("="*60)
    
    print("\n1. 模型总是预测类别 0（无事件）")
    print("   原因:")
    print("   - 数据不平衡：'无事件'帧远多于'有事件'帧")
    print("   - 模型学习到总是预测'无事件'的策略")
    print("   - 损失函数没有给事件类别足够权重")
    
    print("\n2. coarse_scores 的类别 0 总是大于类别 1")
    print("   检查方法:")
    print("   - 查看评估时的 coarse_scores 分布")
    print("   - 如果 coarse_scores[:, 0] > coarse_scores[:, 1] 总是成立")
    print("   - 说明模型倾向于预测类别 0")
    
    print("\n3. 模型没有学习到有效特征")
    print("   可能原因:")
    print("   - 骨架数据质量不好")
    print("   - 特征提取层没有正确学习")
    print("   - 学习率太小或太大")
    
    print("\n4. 损失函数问题")
    print("   检查:")
    print("   - coarse_loss 是否正常（没有 NaN/Inf）")
    print("   - fine_loss 是否正常")
    print("   - 损失值是否合理（不应该太小）")
    
    print("\n" + "="*60)
    print("解决方案:")
    print("="*60)
    
    print("\n1. 使用加权损失函数（推荐）")
    print("   在 MD-FED/train_MD-FED.py 中修改:")
    print("   ```python")
    print("   # 给事件类别更高权重")
    print("   class_weights = torch.tensor([1.0, 10.0]).to(device)")
    print("   coarse_loss = F.cross_entropy(..., weight=class_weights)")
    print("   ```")
    
    print("\n2. 检查数据平衡")
    print("   运行:")
    print("   python -c \"")
    print("   import json")
    print("   with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:")
    print("       data = json.load(f)")
    print("   events = sum(len(v.get('events', [])) for v in data)")
    print("   total_frames = sum(v.get('num_frames', 0) for v in data)")
    print("   print(f'Event frames: {events}/{total_frames} ({events/total_frames*100:.2f}%)')")
    print("   \"")
    
    print("\n3. 降低学习率或增加训练轮数")
    print("   尝试:")
    print("   --learning_rate 0.0001")
    print("   --num_epochs 100")
    
    print("\n4. 检查模型输出")
    print("   在评估函数中添加调试，查看 coarse_scores 的分布")
    
    print("\n" + "="*60)
    print("立即检查:")
    print("="*60)
    print("\n运行以下命令检查数据平衡:")
    print("python check_data_balance.py md_fed_data/f3set-tennis-sub/train.json")


if __name__ == '__main__':
    import os
    
    loss_file = 'MD-FED/md_fed_outputs/stage1/loss.json'
    output_dir = 'MD-FED/md_fed_outputs/stage1'
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        loss_file = os.path.join(output_dir, 'loss.json')
    
    diagnose_no_predictions(loss_file, output_dir)
