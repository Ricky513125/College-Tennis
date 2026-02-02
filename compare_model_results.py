#!/usr/bin/env python3
"""
对比 F3Set 和 Stage 3 模型的评估结果
"""

import json
import sys

def compare_models(f3set_results_file, stage3_results_file, output_file=None):
    """
    对比两个模型的结果
    """
    print(f'\n{"="*60}')
    print('Model Comparison: F3Set vs Stage 3')
    print(f'{"="*60}\n')
    
    with open(f3set_results_file, 'r', encoding='utf-8') as f:
        f3set_results = json.load(f)
    
    with open(stage3_results_file, 'r', encoding='utf-8') as f:
        stage3_results = json.load(f)
    
    print(f"{'Metric':<20} {'F3Set':<15} {'Stage 3':<15} {'Difference':<15} {'Winner':<10} {'Improvement':<12}")
    print('-' * 90)
    
    metrics = ['f1_lcl', 'f1_event', 'f1_element', 'edit_score']
    metric_names = ['F1 (LCL)', 'F1 (event)', 'F1 (element)', 'Edit score']
    
    comparison = {}
    total_improvement = 0
    
    for metric, name in zip(metrics, metric_names):
        f3set_val = f3set_results.get(metric, 0)
        stage3_val = stage3_results.get(metric, 0)
        diff = stage3_val - f3set_val
        winner = 'Stage 3' if diff > 0 else 'F3Set' if diff < 0 else 'Tie'
        
        # 计算改进百分比
        if f3set_val > 0:
            improvement_pct = (diff / f3set_val) * 100
        else:
            improvement_pct = float('inf') if diff > 0 else 0.0
        
        improvement_str = f"{improvement_pct:+.1f}%" if improvement_pct != float('inf') else "N/A"
        
        print(f"{name:<20} {f3set_val:<15.6f} {stage3_val:<15.6f} {diff:+.6f}        {winner:<10} {improvement_str:<12}")
        
        comparison[metric] = {
            'f3set': f3set_val,
            'stage3': stage3_val,
            'difference': diff,
            'improvement_pct': improvement_pct if improvement_pct != float('inf') else None,
            'winner': winner
        }
        
        if improvement_pct != float('inf'):
            total_improvement += improvement_pct
    
    print(f'\n{"="*60}')
    print('Summary')
    print(f'{"="*60}')
    
    # 统计获胜者
    stage3_wins = sum(1 for m in metrics if comparison[m]['winner'] == 'Stage 3')
    f3set_wins = sum(1 for m in metrics if comparison[m]['winner'] == 'F3Set')
    
    print(f'Stage 3 wins: {stage3_wins}/{len(metrics)} metrics')
    print(f'F3Set wins: {f3set_wins}/{len(metrics)} metrics')
    print(f'Average improvement: {total_improvement/len(metrics):.1f}%')
    
    # 详细分析
    print(f'\n{"="*60}')
    print('Detailed Analysis')
    print(f'{"="*60}\n')
    
    print('1. Event Localization (F1 LCL):')
    print(f'   - F3Set: {f3set_results["f1_lcl"]:.1%} (能检测到 {f3set_results["f1_lcl"]*100:.1f}% 的事件位置)')
    print(f'   - Stage 3: {stage3_results["f1_lcl"]:.1%} (能检测到 {stage3_results["f1_lcl"]*100:.1f}% 的事件位置)')
    if comparison['f1_lcl']['winner'] == 'Stage 3':
        print(f'   ✓ Stage 3 在事件定位上更好，提升了 {comparison["f1_lcl"]["improvement_pct"]:.1f}%')
    else:
        print(f'   ⚠️  F3Set 在事件定位上更好')
    
    print('\n2. Event Sequence Matching (F1 event):')
    print(f'   - F3Set: {f3set_results["f1_event"]:.1%} (完整事件序列匹配度)')
    print(f'   - Stage 3: {stage3_results["f1_event"]:.1%} (完整事件序列匹配度)')
    if comparison['f1_event']['winner'] == 'Stage 3':
        print(f'   ✓ Stage 3 在事件序列匹配上更好，提升了 {comparison["f1_event"]["improvement_pct"]:.1f}%')
    else:
        print(f'   ⚠️  F3Set 在事件序列匹配上更好')
    
    print('\n3. Fine-grained Classification (F1 element):')
    print(f'   - F3Set: {f3set_results["f1_element"]:.1%} (细粒度动作元素识别准确度)')
    print(f'   - Stage 3: {stage3_results["f1_element"]:.1%} (细粒度动作元素识别准确度)')
    if comparison['f1_element']['winner'] == 'Stage 3':
        print(f'   ✓ Stage 3 在细粒度分类上更好，提升了 {comparison["f1_element"]["improvement_pct"]:.1f}%')
    else:
        print(f'   ⚠️  F3Set 在细粒度分类上更好')
    
    print('\n4. Sequence Similarity (Edit score):')
    print(f'   - F3Set: {f3set_results["edit_score"]:.2f}/100 (序列相似度)')
    print(f'   - Stage 3: {stage3_results["edit_score"]:.2f}/100 (序列相似度)')
    if comparison['edit_score']['winner'] == 'Stage 3':
        print(f'   ✓ Stage 3 在序列相似度上更好，提升了 {comparison["edit_score"]["improvement_pct"]:.1f}%')
    else:
        print(f'   ⚠️  F3Set 在序列相似度上更好')
    
    print(f'\n{"="*60}\n')
    
    if output_file:
        comparison['summary'] = {
            'stage3_wins': stage3_wins,
            'f3set_wins': f3set_wins,
            'average_improvement': total_improvement / len(metrics)
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"Comparison saved to: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_model_results.py <f3set_results.json> <stage3_results.json> [output.json]")
        sys.exit(1)
    
    f3set_file = sys.argv[1]
    stage3_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'model_comparison.json'
    
    compare_models(f3set_file, stage3_file, output_file)
