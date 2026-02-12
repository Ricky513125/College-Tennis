#!/usr/bin/env python3
"""
对每个视频单独进行预测和评估
使用训练好的 Stage 3 模型分别评估每个视频的性能
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

# Import train_MD-FED.py
import importlib.util
train_md_fed_path = os.path.join(md_fed_dir, 'train_MD-FED.py')
spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
train_MD_FED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_MD_FED)

MD_FED = train_MD_FED.MD_FED
evaluate = train_MD_FED.evaluate
get_best_epoch_and_history = train_MD_FED.get_best_epoch_and_history

from dataset.input_process import ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json


def group_videos_by_id(annotations):
    """按视频ID分组"""
    video_groups = defaultdict(list)
    
    for item in annotations:
        video_name = item.get('video', '')
        if '/' in video_name:
            video_id = video_name.split('/')[0]
        else:
            video_id = video_name
        
        video_groups[video_id].append(item)
    
    return video_groups


def evaluate_single_video_group(model, video_group, video_id, classes, args, window):
    """评估单个视频组（一个视频ID的所有rallies）"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {video_id}")
    print(f"{'='*80}")
    print(f"Number of rallies: {len(video_group)}")
    
    # 创建临时 annotation 文件
    temp_dir = Path(args.temp_dir) / video_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_json = temp_dir / 'test.json'
    
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(video_group, f, indent=2, ensure_ascii=False)
    
    # 创建数据集
    dataset_kwargs = {
        'crop_dim': args.crop_dim,
        'stride': 2
    }
    
    test_data = ActionSeqVideoDataset(
        classes, str(temp_json),
        args.frame_dir, args.clip_len, overlap_len=0,
        num_samples=-1,
        flow_dir=args.flow_dir, pose_dir=None,
        **dataset_kwargs
    )
    
    print(f"Total frames to process: {sum([item['num_frames'] for item in video_group])}")
    
    # 评估
    edit_score = evaluate(
        model, test_data, classes,
        delta=args.delta, window=window, dataset_name=args.dataset_name
    )
    
    print(f"\n{'='*80}")
    print(f"Results for {video_id}:")
    print(f"Edit Score: {edit_score:.4f}")
    print(f"{'='*80}\n")
    
    return {
        'video_id': video_id,
        'num_rallies': len(video_group),
        'num_frames': sum([item['num_frames'] for item in video_group]),
        'edit_score': edit_score
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate each video separately using trained Stage 3 model'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing Stage 3 checkpoints'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch to load (default: best epoch)'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        default='manual_annotations.json',
        help='Path to manual annotations JSON file'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Directory containing extracted video frames'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        default=None,
        help='Directory containing optical flow files (optional)'
    )
    parser.add_argument(
        '--elements_file',
        type=str,
        default='MD-FED/data/f3set-tennis-sub/elements.txt',
        help='Path to elements.txt file'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Name of the dataset'
    )
    parser.add_argument(
        '--delta',
        type=int,
        default=10,
        help='Time tolerance (frames) for evaluation'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default='./temp_eval',
        help='Temporary directory for per-video files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='per_video_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("Per-Video Evaluation")
    print(f"{'='*80}\n")
    
    # Load config
    print("Loading model configuration...")
    config_file = os.path.join(args.checkpoint_dir, 'config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    config = load_json(config_file)
    
    # Extract parameters from config
    visual_arch = config.get('visual_arch', 'rny002_tsm')
    skeleton_arch = config.get('skeleton_arch', 'stgcn++')
    temporal_arch = config.get('temporal_arch', 'gru')
    args.clip_len = config.get('clip_len', 96)
    args.crop_dim = config.get('crop_dim', 224)
    window = config.get('window', 5)
    stage = config.get('stage', 3)
    
    print(f"Model configuration:")
    print(f"  Visual arch: {visual_arch}")
    print(f"  Temporal arch: {temporal_arch}")
    print(f"  Clip len: {args.clip_len}")
    print(f"  Stage: {stage}")
    
    # Load classes
    print(f"\nLoading classes from: {args.elements_file}")
    classes = load_classes(args.elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Load manual annotations
    print(f"\nLoading annotations from: {args.manual_annotations}")
    with open(args.manual_annotations, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)
    print(f"Total annotations: {len(all_annotations)}")
    
    # Group by video ID
    video_groups = group_videos_by_id(all_annotations)
    print(f"\nNumber of video IDs: {len(video_groups)}")
    for video_id, items in video_groups.items():
        print(f"  {video_id}: {len(items)} rallies")
    
    # Create model
    print(f"\nCreating model...")
    model = MD_FED(
        len(classes),
        visual_arch,
        skeleton_arch,
        temporal_arch,
        clip_len=args.clip_len,
        step=2,
        window=window,
        stage=stage,
        multi_gpu=False
    )
    
    # Load checkpoint
    if args.epoch is None:
        print(f"\nFinding best epoch...")
        losses, best_epoch, best_criterion = get_best_epoch_and_history(
            args.checkpoint_dir, 'edit'
        )
        args.epoch = best_epoch
    
    print(f"Loading checkpoint from epoch {args.epoch}...")
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{args.epoch:03d}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load(checkpoint)
    print("✓ Checkpoint loaded")
    
    # Evaluate each video group
    print(f"\n{'='*80}")
    print("Starting per-video evaluation...")
    print(f"{'='*80}\n")
    
    results = []
    
    for video_id in sorted(video_groups.keys()):
        video_group = video_groups[video_id]
        
        try:
            result = evaluate_single_video_group(
                model, video_group, video_id, classes, args, window
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary of Results")
    print(f"{'='*80}\n")
    
    print(f"{'Video ID':<35} {'Rallies':>10} {'Frames':>12} {'Edit Score':>12}")
    print('-' * 80)
    
    total_rallies = 0
    total_frames = 0
    total_edit = 0
    
    for result in results:
        video_id = result['video_id']
        num_rallies = result['num_rallies']
        num_frames = result['num_frames']
        edit_score = result['edit_score']
        
        print(f"{video_id:<35} {num_rallies:>10} {num_frames:>12,} {edit_score:>12.4f}")
        
        total_rallies += num_rallies
        total_frames += num_frames
        total_edit += edit_score * num_rallies  # Weighted by rallies
    
    print('-' * 80)
    avg_edit = total_edit / total_rallies if total_rallies > 0 else 0
    print(f"{'Average':<35} {total_rallies:>10} {total_frames:>12,} {avg_edit:>12.4f}")
    print('=' * 80)
    
    # Save results
    output_data = {
        'checkpoint_dir': args.checkpoint_dir,
        'epoch': args.epoch,
        'dataset_name': args.dataset_name,
        'delta': args.delta,
        'results': results,
        'summary': {
            'total_rallies': total_rallies,
            'total_frames': total_frames,
            'average_edit_score': float(avg_edit)
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {args.output_file}\n")
    
    # Clean up temp directory
    import shutil
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
        print(f"✓ Cleaned up temporary directory: {args.temp_dir}\n")


if __name__ == '__main__':
    main()
