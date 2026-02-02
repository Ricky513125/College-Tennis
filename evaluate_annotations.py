#!/usr/bin/env python3
"""
评估 annotations.json（f3set 模型标注结果）的效果
计算与 Stage 3 相同的指标，便于对比
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from itertools import groupby

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

from util.dataset import load_classes
from util.io import load_json
from util.eval import edit_score, get_labels_start_end_time


def parse_label_string(label_str, classes):
    """
    解析标签字符串（如 "far_serve_middle"）为类别索引列表
    
    Args:
        label_str: 标签字符串，如 "far_serve_middle" 或 "far_middle_serve_-_-_W_-_in"
        classes: 类别字典 {class_name: index}
    
    Returns:
        list: 类别索引列表，如 [1, 3, 8] 表示 far, serve, middle
    """
    if not label_str or label_str == 'NA':
        return []
    
    # 分割标签字符串
    parts = label_str.split('_')
    
    # 过滤掉空字符串和特殊标记（如 '-', 'in', 'out', 'W', 'T' 等）
    valid_parts = []
    skip_tokens = {'-', 'in', 'out', 'W', 'T', 'DM', 'forced-err', 'unforced-err', 'winner'}
    
    for part in parts:
        part = part.strip()
        if part and part not in skip_tokens:
            # 检查是否是类别名
            if part in classes:
                valid_parts.append(classes[part])
            # 处理一些常见的组合
            elif part == 'deduce':
                # deduce 可能对应 return
                if 'return' in classes:
                    valid_parts.append(classes['return'])
    
    return valid_parts


def label_to_binary_vector(label_str, classes):
    """
    将标签字符串转换为二进制向量
    
    Args:
        label_str: 标签字符串
        classes: 类别字典
    
    Returns:
        np.array: 二进制向量，长度为 len(classes)
    """
    vector = np.zeros(len(classes), dtype=np.int32)
    indices = parse_label_string(label_str, classes)
    for idx in indices:
        if 0 < idx <= len(classes):  # 确保索引有效
            vector[idx - 1] = 1  # 类别索引从1开始
    return vector


def annotations_to_frame_labels(annotations, classes, video_name):
    """
    将 annotations.json 格式转换为帧级别的标签
    
    Args:
        annotations: annotations.json 中的视频条目
        classes: 类别字典
        video_name: 视频名称（用于匹配）
    
    Returns:
        tuple: (coarse_label, fine_label)
            - coarse_label: [num_frames] 事件/非事件 (0/1)
            - fine_label: [num_frames, num_classes] 细粒度标签
    """
    # 找到匹配的视频
    video_data = None
    for item in annotations:
        if item.get('video') == video_name:
            video_data = item
            break
    
    if video_data is None:
        return None, None
    
    num_frames = video_data.get('num_frames', 0)
    if num_frames == 0:
        return None, None
    
    # 初始化标签
    coarse_label = np.zeros(num_frames, dtype=np.int32)
    fine_label = np.zeros((num_frames, len(classes)), dtype=np.int32)
    
    # 处理事件
    for event in video_data.get('events', []):
        frame = event.get('frame', -1)
        label_str = event.get('label', '')
        
        if 0 <= frame < num_frames and label_str:
            # 设置粗粒度标签（有事件）
            coarse_label[frame] = 1
            
            # 设置细粒度标签
            fine_vector = label_to_binary_vector(label_str, classes)
            fine_label[frame] = fine_vector
    
    return coarse_label, fine_label


def evaluate_annotations(
    pred_annotations_file,
    gt_annotations_file,
    elements_file,
    delta=10,
    output_file=None
):
    """
    评估 annotations.json 的效果
    
    Args:
        pred_annotations_file: 预测的 annotations.json 文件路径
        gt_annotations_file: 真实标签的 annotations.json 文件路径（如 manual_annotations.json）
        elements_file: elements.txt 文件路径
        delta: 时间容差（帧数）
        output_file: 输出结果文件路径（可选）
    """
    print(f'\n{"="*60}')
    print('Evaluating Annotations')
    print(f'{"="*60}\n')
    
    # 加载数据
    print(f"Loading predictions from: {pred_annotations_file}")
    with open(pred_annotations_file, 'r', encoding='utf-8') as f:
        pred_annotations = json.load(f)
    
    print(f"Loading ground truth from: {gt_annotations_file}")
    with open(gt_annotations_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.load(f)
    
    print(f"Loading classes from: {elements_file}")
    classes = load_classes(elements_file)
    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'
    
    print(f"Found {len(classes)} classes")
    print(f"Prediction videos: {len(pred_annotations)}")
    print(f"Ground truth videos: {len(gt_annotations)}\n")
    
    # 创建视频名称映射
    pred_videos = {item['video']: item for item in pred_annotations}
    gt_videos = {item['video']: item for item in gt_annotations}
    
    # 找到共同的视频
    common_videos = set(pred_videos.keys()) & set(gt_videos.keys())
    print(f"Common videos: {len(common_videos)}\n")
    
    if len(common_videos) == 0:
        print("⚠️  No common videos found! Check video names.")
        return
    
    # 初始化评估指标
    f1_lcl = np.zeros((1, 3), int)  # [tp, fp, fn]
    f1_element = np.zeros((len(classes), 3), int)  # [tp, fp, fn] for each class
    f1_event = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn] for each event
    edit_scores = []
    
    # 打开错误序列文件
    error_file = output_file.replace('.json', '_errors.txt') if output_file else 'annotation_errors.txt'
    f = open(error_file, 'w', encoding='utf-8')
    
    # 评估每个视频
    for video_name in sorted(common_videos):
        # 获取预测标签
        pred_coarse, pred_fine = annotations_to_frame_labels(
            pred_annotations, classes, video_name
        )
        
        # 获取真实标签
        gt_coarse, gt_fine = annotations_to_frame_labels(
            gt_annotations, classes, video_name
        )
        
        if pred_coarse is None or gt_coarse is None:
            print(f"⚠️  Skipping {video_name}: missing labels")
            continue
        
        # 确保长度一致
        min_len = min(len(pred_coarse), len(gt_coarse))
        pred_coarse = pred_coarse[:min_len]
        pred_fine = pred_fine[:min_len, :]
        gt_coarse = gt_coarse[:min_len]
        gt_fine = gt_fine[:min_len, :]
        
        # 计算粗粒度预测（事件/非事件）
        coarse_pred = pred_coarse.copy()
        
        # 计算细粒度预测
        fine_pred = np.zeros_like(pred_fine, dtype=np.int32)
        for i in range(len(pred_fine)):
            if pred_coarse[i] == 1:  # 只在事件帧预测细粒度
                # 使用阈值或 argmax
                fine_pred[i] = (pred_fine[i] > 0.3).astype(np.int32)
        
        # 应用数据集特定的后处理规则（类似 evaluate 函数）
        dataset_name = 'ncaa-rally'  # 或从参数传入
        if 'f3set-tennis-sub' in dataset_name or 'ncaa-rally' in dataset_name:
            for i in range(len(fine_pred)):
                # 处理 serve 和 return
                for start, end in [[0, 2], [2, 5]]:
                    max_idx = np.argmax(pred_fine[i, start:end])
                    fine_pred[i, start + max_idx] = 1
                
                # 处理 approach
                if pred_fine[i, 13] > 0.5:
                    fine_pred[i, 13] = 1
                
                # 如果不是 serve，处理其他动作
                if fine_pred[i, 2] != 1:
                    for start, end in [[5, 7], [7, 13]]:
                        max_idx = np.argmax(pred_fine[i, start:end])
                        fine_pred[i, start + max_idx] = 1
        
        fine_pred = coarse_pred[:, np.newaxis] * fine_pred
        
        # 计算 F1 (LCL) - 事件定位
        for i in range(len(coarse_pred)):
            if coarse_pred[i] == 1 and np.sum(gt_coarse[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 1:
                f1_lcl[0, 0] += 1  # tp
            if coarse_pred[i] == 1 and np.sum(gt_coarse[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 1] += 1  # fp
            if gt_coarse[i] == 1 and np.sum(coarse_pred[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 2] += 1  # fn
        
        # 计算 F1 (element) - 元素级别
        for i in range(len(fine_pred)):
            for j in range(len(fine_pred[0])):
                if fine_pred[i, j] == 1 and np.sum(gt_fine[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
                    f1_element[j, 0] += 1  # tp
                if fine_pred[i, j] == 1 and np.sum(gt_fine[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 1] += 1  # fp
                if gt_fine[i, j] == 1 and np.sum(fine_pred[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 2] += 1  # fn
        
        # 准备序列用于 Edit score
        labels = [int(''.join(str(x) for x in row), 2) for row in gt_fine]
        preds = [int(''.join(str(x) for x in row), 2) for row in fine_pred]
        preds = coarse_pred * preds
        
        # 计算 F1 (event) - 事件级别
        for i in range(len(preds)):
            if preds[i] > 0 and preds[i] in labels[max(0, i - delta):min(len(preds), i + delta + 1)]:
                f1_event[preds[i]][0] += 1  # tp
            if preds[i] > 0 and np.sum(labels[max(0, i - delta):min(len(preds), i + delta + 1)]) == 0:
                f1_event[preds[i]][1] += 1  # fp
            if labels[i] > 0 and labels[i] not in preds[max(0, i - delta):min(len(preds), i + delta + 1)]:
                f1_event[labels[i]][2] += 1  # fn
        
        # 计算 Edit score
        gt = [k for k, g in groupby(labels) if k != 0]
        pred = [k for k, g in groupby(preds) if k != 0]
        edit_scores.append(edit_score(pred, gt))
        
        # 记录错误序列
        print_preds = []
        print_gts = []
        for i in range(len(fine_pred)):
            if gt_coarse[i] == 1:
                print_gt = []
                for j in range(len(fine_pred[0])):
                    if gt_fine[i, j] == 1:
                        print_gt.append(classes_inv.get(j + 1, f'class_{j+1}'))
                print_gts.append('_'.join(print_gt) if print_gt else 'NA')
            if coarse_pred[i] == 1:
                print_pred = []
                for j in range(len(fine_pred[0])):
                    if fine_pred[i, j] == 1:
                        print_pred.append(classes_inv.get(j + 1, f'class_{j+1}'))
                print_preds.append('_'.join(print_pred) if print_pred else 'NA')
        
        pred_sequence = '->'.join(print_preds) if print_preds else '(empty)'
        gt_sequence = '->'.join(print_gts) if print_gts else '(empty)'
        
        if pred_sequence != gt_sequence:
            f.write(f'{video_name}\n')
            f.write(f'{pred_sequence}\n')
            f.write(f'\n{gt_sequence}\n')
            f.write('\n------------------------\n')
    
    f.close()
    
    # 计算最终指标
    precision = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 1] + 1e-10)
    recall = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 2] + 1e-10)
    f1_lcl_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    f1_event_scores = []
    for value in f1_event.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_event_scores.append(f1)
    
    f1_event_mean = np.mean(f1_event_scores) if f1_event_scores else 0.0
    
    precision = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 1] + 1e-10)
    recall = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 2] + 1e-10)
    f1_element_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    edit_score_mean = np.mean(edit_scores) if edit_scores else 0.0
    
    # 打印结果
    print(f'\n{"="*60}')
    print('Evaluation Results')
    print(f'{"="*60}\n')
    print(f'Mean F1 (LCL): {np.mean(f1_lcl_score):.6f}')
    print(f'Mean F1 (event): {f1_event_mean:.6f}')
    print(f'Mean F1 (element): {np.mean(f1_element_score):.6f}')
    print(f'Edit score: {edit_score_mean:.6f}')
    print(f'\n{"="*60}\n')
    
    # 保存结果
    if output_file:
        results = {
            'f1_lcl': float(np.mean(f1_lcl_score)),
            'f1_event': float(f1_event_mean),
            'f1_element': float(np.mean(f1_element_score)),
            'edit_score': float(edit_score_mean),
            'num_videos': len(common_videos),
            'delta': delta,
            'pred_file': pred_annotations_file,
            'gt_file': gt_annotations_file
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
        print(f"Error sequences saved to: {error_file}")
    
    return {
        'f1_lcl': np.mean(f1_lcl_score),
        'f1_event': f1_event_mean,
        'f1_element': np.mean(f1_element_score),
        'edit_score': edit_score_mean
    }


def compare_models(f3set_results_file, stage3_results_file, output_file=None):
    """
    对比两个模型的结果
    
    Args:
        f3set_results_file: f3set 模型的结果文件
        stage3_results_file: Stage 3 模型的结果文件
        output_file: 输出对比文件路径（可选）
    """
    print(f'\n{"="*60}')
    print('Model Comparison')
    print(f'{"="*60}\n')
    
    with open(f3set_results_file, 'r', encoding='utf-8') as f:
        f3set_results = json.load(f)
    
    with open(stage3_results_file, 'r', encoding='utf-8') as f:
        stage3_results = json.load(f)
    
    print(f"{'Metric':<20} {'F3Set':<15} {'Stage 3':<15} {'Difference':<15} {'Winner':<10}")
    print('-' * 75)
    
    metrics = ['f1_lcl', 'f1_event', 'f1_element', 'edit_score']
    metric_names = ['F1 (LCL)', 'F1 (event)', 'F1 (element)', 'Edit score']
    
    comparison = {}
    for metric, name in zip(metrics, metric_names):
        f3set_val = f3set_results.get(metric, 0)
        stage3_val = stage3_results.get(metric, 0)
        diff = stage3_val - f3set_val
        winner = 'Stage 3' if diff > 0 else 'F3Set' if diff < 0 else 'Tie'
        
        print(f"{name:<20} {f3set_val:<15.6f} {stage3_val:<15.6f} {diff:+.6f}        {winner:<10}")
        
        comparison[metric] = {
            'f3set': f3set_val,
            'stage3': stage3_val,
            'difference': diff,
            'winner': winner
        }
    
    print(f'\n{"="*60}\n')
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"Comparison saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate annotations.json and compare with Stage 3 results'
    )
    parser.add_argument(
        '--pred_annotations',
        type=str,
        required=True,
        help='Path to prediction annotations.json (f3set model output)'
    )
    parser.add_argument(
        '--gt_annotations',
        type=str,
        required=True,
        help='Path to ground truth annotations.json (manual_annotations.json)'
    )
    parser.add_argument(
        '--elements_file',
        type=str,
        default='MD-FED/data/f3set-tennis-sub/elements.txt',
        help='Path to elements.txt file'
    )
    parser.add_argument(
        '--delta',
        type=int,
        default=10,
        help='Time tolerance (frames) for F1 calculation (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for results (default: f3set_evaluation_results.json)'
    )
    parser.add_argument(
        '--compare_with',
        type=str,
        default=None,
        help='Path to Stage 3 results file for comparison'
    )
    
    args = parser.parse_args()
    
    # 设置默认输出文件
    if args.output is None:
        args.output = 'f3set_evaluation_results.json'
    
    # 评估
    results = evaluate_annotations(
        args.pred_annotations,
        args.gt_annotations,
        args.elements_file,
        delta=args.delta,
        output_file=args.output
    )
    
    # 对比（如果提供了 Stage 3 结果）
    if args.compare_with:
        compare_models(args.output, args.compare_with, 'model_comparison.json')


if __name__ == '__main__':
    main()
