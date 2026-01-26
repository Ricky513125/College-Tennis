#!/usr/bin/env python3
"""
分析 Stage 1 训练结果，诊断问题并提供改进建议
"""

import json
import os
import argparse
from collections import Counter
import numpy as np


def analyze_annotations(annotations_file):
    """分析标注数据，检查数据质量和分布"""
    print("=" * 60)
    print("1. 数据质量分析")
    print("=" * 60)
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"总 rally 数量: {len(annotations)}")
    
    # 统计事件分布
    event_types = []
    event_counts_per_rally = []
    frame_counts = []
    
    for ann in annotations:
        num_frames = ann.get('num_frames', 0)
        events = ann.get('events', [])
        event_counts_per_rally.append(len(events))
        frame_counts.append(num_frames)
        
        for event in events:
            event_types.append(event.get('label', 'unknown'))
    
    print(f"\n帧数统计:")
    print(f"  平均帧数: {np.mean(frame_counts):.1f}")
    print(f"  最小帧数: {min(frame_counts)}")
    print(f"  最大帧数: {max(frame_counts)}")
    
    print(f"\n每个 rally 的事件数统计:")
    print(f"  平均事件数: {np.mean(event_counts_per_rally):.1f}")
    print(f"  最小事件数: {min(event_counts_per_rally)}")
    print(f"  最大事件数: {max(event_counts_per_rally)}")
    print(f"  无事件 rally 数: {sum(1 for x in event_counts_per_rally if x == 0)}")
    
    # 事件类型分布
    event_counter = Counter(event_types)
    print(f"\n事件类型分布 (Top 10):")
    for event_type, count in event_counter.most_common(10):
        print(f"  {event_type}: {count}")
    
    # 检查事件密度
    event_densities = []
    for ann in annotations:
        num_frames = ann.get('num_frames', 1)
        num_events = len(ann.get('events', []))
        density = num_events / num_frames if num_frames > 0 else 0
        event_densities.append(density)
    
    print(f"\n事件密度 (事件数/帧数):")
    print(f"  平均: {np.mean(event_densities):.4f}")
    print(f"  中位数: {np.median(event_densities):.4f}")
    
    # 检查类别不平衡
    total_frames = sum(frame_counts)
    total_events = sum(event_counts_per_rally)
    event_frame_ratio = total_events / total_frames if total_frames > 0 else 0
    
    print(f"\n类别不平衡分析:")
    print(f"  总帧数: {total_frames}")
    print(f"  总事件数: {total_events}")
    print(f"  事件帧比例: {event_frame_ratio:.4f} ({event_frame_ratio*100:.2f}%)")
    print(f"  非事件帧比例: {(1-event_frame_ratio):.4f} ({(1-event_frame_ratio)*100:.2f}%)")
    print(f"  不平衡比例: {(1-event_frame_ratio)/event_frame_ratio:.1f}:1" if event_frame_ratio > 0 else "  N/A")
    
    return {
        'num_rallies': len(annotations),
        'avg_frames': np.mean(frame_counts),
        'avg_events_per_rally': np.mean(event_counts_per_rally),
        'event_frame_ratio': event_frame_ratio,
        'event_types': event_counter
    }


def analyze_training_config(output_dir):
    """分析训练配置"""
    print("\n" + "=" * 60)
    print("2. 训练配置分析")
    print("=" * 60)
    
    config_file = os.path.join(output_dir, 'config.txt')
    history_file = os.path.join(output_dir, 'history.json')
    
    if os.path.exists(config_file):
        print("\n训练配置:")
        with open(config_file, 'r') as f:
            print(f.read())
    else:
        print(f"配置文件未找到: {config_file}")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print("\n训练历史:")
        if history:
            train_losses = [h.get('train', 0) for h in history]
            val_losses = [h.get('val', 0) for h in history]
            
            print(f"  总 epoch 数: {len(history)}")
            print(f"  初始训练损失: {train_losses[0]:.4f}")
            print(f"  最终训练损失: {train_losses[-1]:.4f}")
            print(f"  初始验证损失: {val_losses[0]:.4f}")
            print(f"  最终验证损失: {val_losses[-1]:.4f}")
            
            if len(val_losses) > 1:
                loss_reduction = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
                print(f"  验证损失下降: {loss_reduction:.2f}%")
                
                # 检查是否过拟合
                if train_losses[-1] < val_losses[-1] * 0.5:
                    print(f"  ⚠️  警告: 可能存在过拟合 (训练损失远小于验证损失)")
                
                # 检查损失是否还在下降
                if len(val_losses) >= 5:
                    recent_trend = np.mean(val_losses[-5:]) - np.mean(val_losses[-10:-5]) if len(val_losses) >= 10 else 0
                    if recent_trend > 0:
                        print(f"  ⚠️  警告: 最近验证损失在上升，可能需要早停")
    else:
        print(f"历史文件未找到: {history_file}")


def provide_recommendations(data_stats):
    """提供改进建议"""
    print("\n" + "=" * 60)
    print("3. 改进建议")
    print("=" * 60)
    
    recommendations = []
    
    # 检查类别不平衡
    if data_stats['event_frame_ratio'] < 0.01:
        recommendations.append({
            'issue': '严重的类别不平衡',
            'severity': 'high',
            'description': f"事件帧比例只有 {data_stats['event_frame_ratio']*100:.2f}%，模型很难学习到事件",
            'solutions': [
                '增加类别权重（已在代码中设置为20.0，可以进一步增加到30.0-50.0）',
                '使用 focal loss 替代 cross-entropy loss',
                '使用数据增强增加事件帧的采样',
                '使用 class-balanced sampling'
            ]
        })
    
    # 检查事件密度
    if data_stats['avg_events_per_rally'] < 2:
        recommendations.append({
            'issue': '每个 rally 的事件数太少',
            'severity': 'medium',
            'description': f"平均每个 rally 只有 {data_stats['avg_events_per_rally']:.1f} 个事件",
            'solutions': [
                '检查标注是否完整',
                '考虑使用更长的 clip_len 来捕获更多事件',
                '减少 stride 来增加采样密度'
            ]
        })
    
    # 检查数据量
    if data_stats['num_rallies'] < 100:
        recommendations.append({
            'issue': '训练数据量可能不足',
            'severity': 'medium',
            'description': f"只有 {data_stats['num_rallies']} 个 rally，可能不足以训练模型",
            'solutions': [
                '增加训练数据',
                '使用数据增强',
                '使用预训练模型',
                '减少模型复杂度'
            ]
        })
    
    # 通用建议
    recommendations.append({
        'issue': '模型只预测第一个事件',
        'severity': 'high',
        'description': '从结果看，模型主要只预测了第一个事件（near_serve），没有预测后续事件',
        'solutions': [
            '增加训练轮数（当前可能训练不充分）',
            '调整学习率（可能需要更小的学习率进行精细调优）',
            '使用学习率调度器（cosine annealing 或 step decay）',
            '增加 clip_len 以捕获更长的事件序列',
            '检查 skeleton 数据质量（Stage 1 只使用 skeleton）',
            '使用 focal loss 来处理类别不平衡',
            '增加梯度累积步数以稳定训练'
        ]
    })
    
    # 打印建议
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']} ({rec['severity']} severity)")
        print(f"   问题: {rec['description']}")
        print(f"   解决方案:")
        for solution in rec['solutions']:
            print(f"     - {solution}")
    
    return recommendations


def generate_improved_config(output_dir, recommendations):
    """生成改进的训练配置"""
    print("\n" + "=" * 60)
    print("4. 改进的训练配置建议")
    print("=" * 60)
    
    print("\n建议的训练参数:")
    print("""
# 改进的 Stage 1 训练配置

python run_md_fed_stage1.py \\
    --data_dir md_fed_data \\
    --dataset f3set-tennis-sub \\
    --pose_dir /path/to/skeleton/pkl/files \\
    --output_dir ./md_fed_outputs/stage1_improved \\
    --num_epochs 100 \\          # 增加训练轮数
    --batch_size 4 \\
    --learning_rate 0.0005 \\    # 降低学习率
    --clip_len 128 \\            # 增加 clip 长度以捕获更多事件
    --stride 1 \\                # 减少 stride 增加采样密度
    --warm_up_epochs 5 \\        # 增加 warmup
    --acc_grad_iter 2            # 增加梯度累积
    """)
    
    # 保存改进建议到文件
    suggestions_file = os.path.join(output_dir, 'improvement_suggestions.txt')
    with open(suggestions_file, 'w', encoding='utf-8') as f:
        f.write("Stage 1 训练改进建议\n")
        f.write("=" * 60 + "\n\n")
        for rec in recommendations:
            f.write(f"{rec['issue']} ({rec['severity']} severity)\n")
            f.write(f"问题: {rec['description']}\n")
            f.write("解决方案:\n")
            for solution in rec['solutions']:
                f.write(f"  - {solution}\n")
            f.write("\n")
    
    print(f"\n改进建议已保存到: {suggestions_file}")


def main():
    parser = argparse.ArgumentParser(
        description='分析 Stage 1 训练结果并提供改进建议'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='./manual_annotations.json',
        help='标注文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./md_fed_outputs/stage1',
        help='Stage 1 训练输出目录'
    )
    
    args = parser.parse_args()
    
    # 分析数据
    data_stats = analyze_annotations(args.annotations)
    
    # 分析训练配置
    analyze_training_config(args.output_dir)
    
    # 提供建议
    recommendations = provide_recommendations(data_stats)
    
    # 生成改进配置
    generate_improved_config(args.output_dir, recommendations)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
