#!/usr/bin/env python3
"""
分析 error_sequences.txt 的格式，理解预测和真实标签的对应关系
"""

import sys
import re

def analyze_error_format(error_file):
    """分析错误序列格式"""
    
    print("="*60)
    print("分析 error_sequences.txt 格式")
    print("="*60)
    
    with open(error_file, 'r') as f:
        content = f.read()
    
    # 解析格式
    # 格式应该是：
    # 视频名
    # 预测序列 (用 -> 连接)
    # 空行
    # 真实标签序列 (用 -> 连接)
    # ------------------------
    
    errors = []
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == '------------------------':
            i += 1
            continue
        
        if not line:
            i += 1
            continue
        
        # 视频名（通常包含路径或下划线）
        if '/' in line or '_' in line or line.endswith('.mp4'):
            video = line
            i += 1
            
            # 预测序列
            pred_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            # 跳过空行
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # 真实标签序列
            gt_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            errors.append({
                'video': video,
                'pred': pred_line,
                'gt': gt_line
            })
        else:
            i += 1
    
    print(f"\n找到 {len(errors)} 个错误序列\n")
    
    # 分析
    pred_empty = sum(1 for e in errors if not e['pred'])
    gt_empty = sum(1 for e in errors if not e['gt'])
    both_have = sum(1 for e in errors if e['pred'] and e['gt'])
    pred_only = sum(1 for e in errors if e['pred'] and not e['gt'])
    gt_only = sum(1 for e in errors if not e['pred'] and e['gt'])
    
    print("="*60)
    print("统计信息:")
    print("="*60)
    print(f"  总错误序列数: {len(errors)}")
    print(f"  预测为空: {pred_empty}")
    print(f"  真实标签为空: {gt_empty}")
    print(f"  两者都有: {both_have}")
    print(f"  只有预测: {pred_only}")
    print(f"  只有真实标签: {gt_only}")
    
    print("\n" + "="*60)
    print("前 5 个示例:")
    print("="*60)
    for i, e in enumerate(errors[:5]):
        print(f"\n示例 {i+1}:")
        print(f"  视频: {e['video'][:60]}...")
        print(f"  预测序列: {e['pred'] or '(空)'}")
        print(f"  真实标签: {e['gt'] or '(空)'}")
        
        if e['pred'] and e['gt']:
            pred_events = e['pred'].split('->')
            gt_events = e['gt'].split('->')
            print(f"  预测事件数: {len(pred_events)}")
            print(f"  真实事件数: {len(gt_events)}")
            if len(pred_events) == len(gt_events):
                print("  事件数量相同，但内容不同")
            else:
                print(f"  事件数量不同: 预测 {len(pred_events)} vs 真实 {len(gt_events)}")
    
    print("\n" + "="*60)
    print("诊断:")
    print("="*60)
    
    if gt_empty > len(errors) * 0.5:
        print("\n⚠️  超过 50% 的真实标签序列为空！")
        print("\n可能原因:")
        print("  1. 这些视频在验证集中没有事件标签")
        print("  2. 标签格式不匹配（coarse_label 全为 0）")
        print("  3. 标签加载有问题")
        print("\n解决方案:")
        print("  1. 检查 md_fed_data/f3set-tennis-sub/val.json 中这些视频是否有 events")
        print("  2. 检查标签格式是否正确")
        print("  3. 确认验证集使用的是正确的文件")
    
    if pred_only > len(errors) * 0.5:
        print("\n⚠️  模型在预测事件，但真实标签为空")
        print("  这意味着模型预测了不存在的事件（False Positive）")
        print("  这会导致 F1=0，因为 TP=0")
    
    print("\n" + "="*60)
    print("理解:")
    print("="*60)
    print("\n1. 预测序列格式:")
    print("   - 每个事件用 '->' 连接")
    print("   - 例如: 'far_serve->near_return_bh_gs'")
    print("   - 表示视频中按时间顺序发生的事件")
    print("\n2. 真实标签序列格式:")
    print("   - 同样用 '->' 连接")
    print("   - 从验证集的 JSON 文件中读取")
    print("\n3. 对比方式:")
    print("   - 使用 Edit Score 对比整个序列")
    print("   - 使用 F1 对比每个事件的位置和类型")
    print("\n4. 如果真实标签为空:")
    print("   - 说明这些视频在验证集中没有标注事件")
    print("   - 或者标签格式转换有问题")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        error_file = 'MD-FED/error_sequences.txt'
    else:
        error_file = sys.argv[1]
    
    analyze_error_format(error_file)
