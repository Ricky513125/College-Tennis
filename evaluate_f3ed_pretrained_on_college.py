#!/usr/bin/env python3
"""
实验1：评估F3ED在F3Set上预训练后，在校园网球数据集上的效果（不微调）

这个实验展示迁移学习的效果：
- 使用F3Set数据集上预训练的F3ED模型
- 直接在校园网球数据集上评估（不进行微调）
- 对比从头训练的效果

Usage:
    python evaluate_f3ed_pretrained_on_college.py \
        --f3set_model_dir ./F3Set/f3set-model/f3ed \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --output_dir ./f3ed_pretrained_evaluation
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add F3Set to path
f3set_dir = os.path.join(os.path.dirname(__file__), 'F3Set')
if os.path.exists(f3set_dir):
    sys.path.insert(0, f3set_dir)

# Import F3ED model
from train_f3set_f3ed import F3Set, evaluate
from dataset.frame_process import ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json, store_json
import re


def get_best_epoch(model_dir, key='val_edit'):
    """Get best epoch from loss.json"""
    loss_file = os.path.join(model_dir, 'loss.json')
    if not os.path.exists(loss_file):
        return None
    
    data = load_json(loss_file)
    if not data:
        return None
    
    best = max(data, key=lambda x: x.get(key, 0))
    return best['epoch']


def get_last_epoch(model_dir):
    """Get last epoch from checkpoint files"""
    regex = re.compile(r'checkpoint_(\d+)\.pt')
    
    last_epoch = -1
    if not os.path.exists(model_dir):
        return None
    
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    
    if last_epoch < 0:
        return None
    return last_epoch


def convert_manual_annotations_to_f3set_format(manual_annotations_file, output_file):
    """
    Convert manual_annotations.json to F3Set format (same as MD-FED format).
    """
    print(f"Converting manual annotations from {manual_annotations_file}...")
    
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)
    
    converted_data = []
    
    for video_data in manual_data:
        video_name = video_data['video']
        num_frames = video_data['num_frames']
        fps = video_data.get('fps', 30.0)
        height = video_data.get('height', 1080)
        width = video_data.get('width', 1920)
        
        events = []
        for event in video_data.get('events', []):
            frame = event['frame']
            label_str = event['label']
            
            event_dict = {
                'frame': frame,
                'label': label_str,
            }
            
            if 'outcome' in event:
                event_dict['outcome'] = event['outcome']
            
            events.append(event_dict)
        
        converted_entry = {
            'fps': fps,
            'height': height,
            'width': width,
            'num_frames': num_frames,
            'video': video_name,
            'far_name': video_data.get('far_name', 'Unknown'),
            'far_hand': video_data.get('far_hand', 'RH'),
            'far_set': video_data.get('far_set', 0),
            'far_game': video_data.get('far_game', 0),
            'far_point': video_data.get('far_point', 0),
            'near_name': video_data.get('near_name', 'Unknown'),
            'near_hand': video_data.get('near_hand', 'RH'),
            'near_set': video_data.get('near_set', 0),
            'near_game': video_data.get('near_game', 0),
            'near_point': video_data.get('near_point', 0),
            'events': events
        }
        
        converted_data.append(converted_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} videos to {output_file}")
    return converted_data


def evaluate_f3ed_pretrained(args):
    """
    评估F3Set预训练的F3ED模型在校园网球数据集上的效果
    """
    print("="*70)
    print("实验1：F3ED在F3Set上训练，在校园网球的效果（不微调）")
    print("="*70)
    
    # Step 1: Load F3Set model config and checkpoint
    print("\nStep 1: Loading F3Set pretrained model...")
    config_path = os.path.join(args.f3set_model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_json(config_path)
    print(f"  Model config: {config_path}")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Feature arch: {config['feature_arch']}")
    print(f"  Temporal arch: {config['temporal_arch']}")
    print(f"  Use CTX: {config.get('use_ctx', False)}")
    
    # Get best epoch
    best_epoch = get_best_epoch(args.f3set_model_dir)
    if best_epoch is None:
        best_epoch = get_last_epoch(args.f3set_model_dir)
    
    if best_epoch is None:
        raise ValueError(f"No checkpoint found in {args.f3set_model_dir}")
    
    print(f"  Best epoch: {best_epoch}")
    
    # Load classes from F3Set (should match college tennis classes)
    # Try multiple locations
    elements_file = None
    possible_locations = [
        os.path.join('F3Set', 'data', 'f3set-tennis', 'elements.txt'),
        os.path.join('MD-FED', 'data', 'f3set-tennis-sub', 'elements.txt'),
        os.path.join('data', 'f3set-tennis', 'elements.txt'),
        'elements.txt',
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            elements_file = loc
            break
    
    if elements_file is None:
        raise FileNotFoundError("Could not find elements.txt in any expected location")
    
    print(f"  Elements file: {elements_file}")
    classes = load_classes(elements_file)
    print(f"  Number of classes: {len(classes)}")
    
    # Load F3ED model
    model = F3Set(
        len(classes),
        config['feature_arch'],
        config['temporal_arch'],
        clip_len=config['clip_len'],
        step=config.get('stride', 2),
        window=config.get('window', 5),
        use_ctx=config.get('use_ctx', True),
        multi_gpu=config.get('gpu_parallel', False)
    )
    
    checkpoint_path = os.path.join(
        args.f3set_model_dir,
        'checkpoint_{:03d}.pt'.format(best_epoch)
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"  Loading checkpoint: {checkpoint_path}")
    model.load(torch.load(checkpoint_path))
    model._model.eval()
    
    # Step 2: Prepare college tennis data
    print("\nStep 2: Preparing college tennis data...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert manual annotations to F3Set format
    converted_json = os.path.join(args.output_dir, 'college_tennis_test.json')
    convert_manual_annotations_to_f3set_format(
        args.manual_annotations,
        converted_json
    )
    
    # Create dataset
    test_dataset = ActionSeqVideoDataset(
        classes,
        converted_json,
        args.frame_dir,
        config['clip_len'],
        overlap_len=config['clip_len'] // 2,
        crop_dim=config.get('crop_dim', 224),
        stride=config.get('stride', 2)
    )
    
    test_dataset.print_info()
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating on college tennis dataset...")
    print("="*70)
    
    # Use F3ED's evaluate function (it prints metrics but only returns edit_score)
    # We'll capture the printed output or use the return value
    edit_score_result = evaluate(model, test_dataset, classes, window=config.get('window', 5))
    
    # Note: F3Set's evaluate function prints Mean F1 (event) and Mean F1 (element)
    # but only returns edit_score. For full metrics, we would need to modify
    # the evaluate function or reimplement it. For now, we save what we have.
    
    # Step 4: Save results
    print("\nStep 4: Saving results...")
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    
    # Convert results to standard format (matching MD-FED evaluation)
    evaluation_results = {
        'experiment': 'F3ED pretrained on F3Set, evaluated on college tennis (no fine-tuning)',
        'f3set_model_dir': args.f3set_model_dir,
        'f3set_best_epoch': best_epoch,
        'college_tennis_annotations': args.manual_annotations,
        'num_videos': len(test_dataset._src_data),
        'edit_score': float(edit_score_result),
        'note': 'F3Set evaluate function prints Mean F1 (event) and Mean F1 (element) to console, but only returns edit_score. Check console output for full metrics.'
    }
    
    store_json(results_file, evaluation_results, pretty=True)
    print(f"  Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {results_file}")
    print("\nThis experiment shows the transfer learning performance:")
    print("  - F3ED model was pretrained on F3Set dataset")
    print("  - Directly evaluated on college tennis dataset (no fine-tuning)")
    print("  - Compare with train_f3ed_from_scratch_on_college.py for ablation")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate F3ED pretrained on F3Set on college tennis dataset'
    )
    parser.add_argument(
        '--f3set_model_dir',
        type=str,
        required=True,
        help='Path to F3Set pretrained F3ED model directory'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        required=True,
        help='Path to manual_annotations.json (college tennis dataset)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Path to frame directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./f3ed_pretrained_evaluation',
        help='Output directory for evaluation results'
    )
    
    args = parser.parse_args()
    evaluate_f3ed_pretrained(args)


if __name__ == '__main__':
    main()
