#!/usr/bin/env python3
"""
对比 VTN 和 MD-FED Stage 3 的评估结果
"""

import json
import argparse
from pathlib import Path


def load_results(file_path):
    """加载结果文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 {file_path}")
        return None


def format_metric(value, is_improvement=None):
    """格式化指标值"""
    formatted = f"{value:.6f}"
    if is_improvement is None:
        return formatted
    elif is_improvement:
        return f"{formatted} ✅"
    else:
        return f"{formatted} ❌"


def compare_results(vtn_file, mdfed_file, output_file=None):
    """
    对比两个模型的结果
    
    Args:
        vtn_file: VTN 结果文件
        mdfed_file: MD-FED 结果文件
        output_file: 输出文件（可选）
    """
    print(f"{'='*80}")
    print("VTN vs MD-FED Stage 3 对比")
    print(f"{'='*80}\n")
    
    # 加载结果
    print(f"加载 VTN 结果: {vtn_file}")
    vtn_results = load_results(vtn_file)
    if vtn_results is None:
        return
    
    print(f"加载 MD-FED 结果: {mdfed_file}")
    mdfed_results = load_results(mdfed_file)
    if mdfed_results is None:
        return
    
    print(f"\n{'='*80}")
    print("评估指标对比")
    print(f"{'='*80}\n")
    
    # 定义要对比的指标
    metrics = {
        'f1_lcl': 'F1 (LCL)',
        'f1_element': 'F1 (element)',
        'f1_event': 'F1 (event)',
        'edit_score': 'Edit Score'
    }
    
    # 打印表头
    print(f"{'指标':<20} {'MD-FED Stage 3':<20} {'VTN':<20} {'差异':<15} {'胜者':<10}")
    print(f"{'─'*85}")
    
    comparison = {}
    total_wins = {'vtn': 0, 'mdfed': 0, 'tie': 0}
    
    for metric_key, metric_name in metrics.items():
        mdfed_val = mdfed_results.get(metric_key, 0.0)
        vtn_val = vtn_results.get(metric_key, 0.0)
        
        diff = vtn_val - mdfed_val
        diff_pct = (diff / mdfed_val * 100) if mdfed_val > 0 else 0
        
        # 判断胜者
        if abs(diff) < 0.001:
            winner = "Tie"
            total_wins['tie'] += 1
            winner_symbol = "="
        elif diff > 0:
            winner = "VTN"
            total_wins['vtn'] += 1
            winner_symbol = "✅"
        else:
            winner = "MD-FED"
            total_wins['mdfed'] += 1
            winner_symbol = "✅"
        
        # 格式化差异
        if diff >= 0:
            diff_str = f"+{diff:.6f} ({diff_pct:+.1f}%)"
        else:
            diff_str = f"{diff:.6f} ({diff_pct:.1f}%)"
        
        print(f"{metric_name:<20} {mdfed_val:<20.6f} {vtn_val:<20.6f} {diff_str:<15} {winner:<10}")
        
        comparison[metric_key] = {
            'mdfed': mdfed_val,
            'vtn': vtn_val,
            'difference': diff,
            'difference_pct': diff_pct,
            'winner': winner
        }
    
    print(f"{'─'*85}")
    
    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}\n")
    
    print(f"总体胜率:")
    print(f"  VTN 胜出: {total_wins['vtn']}/{len(metrics)} 项指标 ({total_wins['vtn']/len(metrics)*100:.1f}%)")
    print(f"  MD-FED 胜出: {total_wins['mdfed']}/{len(metrics)} 项指标 ({total_wins['mdfed']/len(metrics)*100:.1f}%)")
    print(f"  持平: {total_wins['tie']}/{len(metrics)} 项指标")
    
    # 推荐结论
    print(f"\n推荐结论:")
    if total_wins['vtn'] > total_wins['mdfed']:
        print(f"  🏆 VTN 在本数据集上表现更好")
        recommendation = "VTN"
    elif total_wins['mdfed'] > total_wins['vtn']:
        print(f"  🏆 MD-FED Stage 3 在本数据集上表现更好")
        recommendation = "MD-FED"
    else:
        print(f"  ⚖️  两个模型表现相当")
        recommendation = "Tie"
    
    # 额外信息
    print(f"\n模型信息:")
    
    # 从结果中提取信息（如果有）
    vtn_info = {
        '视频数量': vtn_results.get('num_videos', 'N/A'),
        '时间容差': vtn_results.get('delta', 'N/A')
    }
    
    mdfed_info = {
        '视频数量': mdfed_results.get('num_videos', 'N/A'),
        '时间容差': mdfed_results.get('delta', 'N/A')
    }
    
    print(f"  VTN:")
    for k, v in vtn_info.items():
        print(f"    {k}: {v}")
    
    print(f"  MD-FED Stage 3:")
    for k, v in mdfed_info.items():
        print(f"    {k}: {v}")
    
    # 保存结果
    if output_file:
        output_data = {
            'comparison': comparison,
            'summary': {
                'vtn_wins': total_wins['vtn'],
                'mdfed_wins': total_wins['mdfed'],
                'ties': total_wins['tie'],
                'recommendation': recommendation
            },
            'vtn_results': vtn_results,
            'mdfed_results': mdfed_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n详细对比结果已保存到: {output_file}")
    
    print(f"\n{'='*80}\n")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='对比 VTN 和 MD-FED Stage 3 的评估结果'
    )
    parser.add_argument(
        '--vtn_results',
        type=str,
        default='vtn_outputs/comparison/best_model_metrics.json',
        help='VTN 结果文件路径'
    )
    parser.add_argument(
        '--mdfed_results',
        type=str,
        default='MD-FED/md_fed_outputs/stage3/evaluation_results.json',
        help='MD-FED Stage 3 结果文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_comparison_results.json',
        help='输出文件路径'
    )
    
    args = parser.parse_args()
    
    compare_results(args.vtn_results, args.mdfed_results, args.output)


if __name__ == '__main__':
    main()
