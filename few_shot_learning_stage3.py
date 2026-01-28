#!/usr/bin/env python3
"""
Few-shot learning (Stage 3 fine-tuning) using manually verified labeled campus tennis data.
This script fine-tunes the Stage 2 model on a small, manually verified subset.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

# Import train_MD-FED.py using importlib (handles hyphen in filename)
import importlib.util
train_md_fed_path = os.path.join(md_fed_dir, 'train_MD-FED.py')
spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
train_MD_FED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_MD_FED)

# Import from the module
MD_FED = train_MD_FED.MD_FED
evaluate = train_MD_FED.evaluate
get_best_epoch_and_history = train_MD_FED.get_best_epoch_and_history
get_datasets = train_MD_FED.get_datasets
get_lr_scheduler = train_MD_FED.get_lr_scheduler
store_config = train_MD_FED.store_config
get_args = train_MD_FED.get_args

from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json, store_json
from torch.utils.data import DataLoader
import random


def convert_manual_annotations_to_md_fed_format(manual_annotations_file, output_file, elements_file):
    """
    Convert manual_annotations.json to MD-FED format.
    """
    print(f"Converting manual annotations from {manual_annotations_file}...")
    
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)
    
    classes = load_classes(elements_file)
    
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


def prepare_few_shot_data(manual_annotations_file, output_dir, dataset_name='ncaa-rally', train_ratio=0.8):
    """
    Prepare few-shot learning data by splitting manual annotations into train/val sets.
    
    Args:
        manual_annotations_file: Path to manual annotations
        output_dir: Directory to save prepared data
        dataset_name: Name of the dataset
        train_ratio: Ratio of data to use for training (rest for validation)
    """
    print(f"Preparing few-shot learning data...")
    
    # Create output directory
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manual annotations
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total manual annotations: {len(annotations)}")
    
    # Shuffle and split
    random.seed(42)
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]
    
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    
    # Save train.json
    train_file = data_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved train data to {train_file}")
    
    # Save val.json
    val_file = data_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved val data to {val_file}")
    
    # Copy elements.txt from MD-FED if it exists
    elements_src = os.path.join('MD-FED', 'data', 'f3set-tennis-sub', 'elements.txt')
    if not os.path.exists(elements_src):
        elements_src = os.path.join('data', 'f3set-tennis-sub', 'elements.txt')
    
    if os.path.exists(elements_src):
        import shutil
        elements_dst = data_dir / 'elements.txt'
        shutil.copy(elements_src, elements_dst)
        print(f"Copied elements.txt to {elements_dst}")
    else:
        print(f"Warning: elements.txt not found at {elements_src}")
    
    return str(data_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Few-shot learning (Stage 3) using manually verified labeled campus tennis data'
    )
    parser.add_argument(
        '--stage2_checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing Stage 2 checkpoints (e.g., ./MD-FED/md_fed_outputs/stage2)'
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
        '--pose_dir',
        type=str,
        default=None,
        help='Directory containing pose/skeleton files (optional, not used in Stage 3)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save Stage 3 checkpoints (e.g., ./MD-FED/md_fed_outputs/stage3)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Name of the dataset for few-shot learning'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='few_shot_data',
        help='Directory to save prepared few-shot data'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data to use for training (default: 0.8)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001, lower for fine-tuning)'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=None,
        help='Clip length (default: from Stage 2 config)'
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=None,
        help='Crop dimension (default: from Stage 2 config)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=None,
        help='NMS window size (default: from Stage 2 config)'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default=None,
        help='Visual architecture (default: from Stage 2 config)'
    )
    parser.add_argument(
        '--skeleton_arch',
        type=str,
        default=None,
        help='Skeleton architecture (default: from Stage 2 config, not used in Stage 3)'
    )
    parser.add_argument(
        '--temporal_arch',
        type=str,
        default=None,
        help='Temporal architecture (default: from Stage 2 config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("Few-Shot Learning (Stage 3 Fine-tuning)")
    print(f"{'='*60}\n")
    
    # Step 1: Prepare few-shot data
    print("Step 1: Preparing few-shot learning data...")
    data_dir = prepare_few_shot_data(
        args.manual_annotations,
        args.data_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio
    )
    
    # Step 2: Load Stage 2 model config
    print(f"\nStep 2: Loading Stage 2 model configuration...")
    stage2_config_file = os.path.join(args.stage2_checkpoint_dir, 'config.json')
    if not os.path.exists(stage2_config_file):
        raise FileNotFoundError(f"Stage 2 config not found: {stage2_config_file}")
    
    stage2_config = load_json(stage2_config_file)
    
    # Use Stage 2 config values if not specified
    visual_arch = args.visual_arch if args.visual_arch is not None else stage2_config.get('visual_arch', 'rny002_tsm')
    skeleton_arch = args.skeleton_arch if args.skeleton_arch is not None else stage2_config.get('skeleton_arch', 'stgcn++')
    temporal_arch = args.temporal_arch if args.temporal_arch is not None else stage2_config.get('temporal_arch', 'gru')
    clip_len = args.clip_len if args.clip_len is not None else stage2_config.get('clip_len', 96)
    crop_dim = args.crop_dim if args.crop_dim is not None else stage2_config.get('crop_dim', 224)
    window = args.window if args.window is not None else stage2_config.get('window', 5)
    
    print(f"Using configuration:")
    print(f"  Visual arch: {visual_arch}")
    print(f"  Temporal arch: {temporal_arch}")
    print(f"  Clip len: {clip_len}")
    print(f"  Crop dim: {crop_dim}")
    print(f"  Window: {window}")
    
    # Step 3: Load classes
    elements_file = os.path.join(data_dir, 'elements.txt')
    if not os.path.exists(elements_file):
        raise FileNotFoundError(f"Elements file not found: {elements_file}")
    
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Step 4: Create datasets
    print(f"\nStep 3: Creating datasets...")
    
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    # For few-shot learning, use smaller dataset size
    epoch_num_frames = 10000  # Small dataset for few-shot
    dataset_len = epoch_num_frames // (clip_len * 2)  # stride=2
    
    dataset_kwargs = {
        'crop_dim': crop_dim,
        'stride': 2
    }
    
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, clip_len, dataset_len,
        is_eval=False, dilate_len=0, stage=3,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=None,  # Stage 3 doesn't use skeleton
        **dataset_kwargs
    )
    train_data.print_info()
    
    val_data = ActionSeqDataset(
        classes, val_json,
        args.frame_dir, clip_len, dataset_len // 4,
        dilate_len=0, stage=3,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=None,
        **dataset_kwargs
    )
    val_data.print_info()
    
    val_data_frames = ActionSeqVideoDataset(
        classes, val_json,
        args.frame_dir, clip_len, overlap_len=0,
        num_samples=-1,
        flow_dir=args.flow_dir, pose_dir=None,
        **dataset_kwargs
    )
    
    # Step 5: Create model
    print(f"\nStep 4: Creating Stage 3 model...")
    model = MD_FED(
        len(classes),
        visual_arch,
        skeleton_arch,
        temporal_arch,
        clip_len=clip_len,
        step=2,
        window=window,
        stage=3,  # Stage 3 for few-shot learning
        multi_gpu=False
    )
    
    # Step 6: Load Stage 2 checkpoint
    print(f"\nStep 5: Loading Stage 2 checkpoint...")
    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.stage2_checkpoint_dir, 'edit'
    )
    print(f'Loading from Stage 2 epoch {best_epoch}')
    
    stage2_checkpoint = torch.load(
        os.path.join(args.stage2_checkpoint_dir, f'checkpoint_{best_epoch:03d}.pt'),
        map_location=args.device
    )
    model.load(stage2_checkpoint)
    print("✓ Stage 2 checkpoint loaded")
    
    # Step 7: Setup training
    print(f"\nStep 6: Setting up training...")
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    
    train_loader = DataLoader(
        train_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=1
    )
    
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4
    )
    
    num_steps_per_epoch = len(train_loader)
    warm_up_epochs = 3
    cosine_epochs = args.num_epochs - warm_up_epochs
    print(f'Using Linear Warmup ({warm_up_epochs}) + Cosine Annealing LR ({cosine_epochs})')
    lr_scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])
    
    # Step 8: Training loop
    print(f"\nStep 7: Starting training...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Save dir: {args.save_dir}\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    losses = []
    best_epoch = None
    best_edit_score = 0
    
    for epoch in range(args.num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=1)
        val_loss = model.epoch(val_loader, acc_grad_iter=1)
        
        print(f'[Epoch {epoch}] Train loss: {train_loss:.5f} Val loss: {val_loss:.5f}')
        
        # Evaluate on validation set
        val_edit = 0
        if epoch >= args.num_epochs - 10:  # Evaluate in last 10 epochs
            val_edit = evaluate(
                model, val_data_frames, classes,
                delta=10, window=window, dataset_name=args.dataset_name
            )
            if val_edit > best_edit_score:
                best_edit_score = val_edit
                best_epoch = epoch
                print(f'New best epoch! Edit score: {val_edit:.4f}')
        
        losses.append({
            'epoch': epoch,
            'train': train_loss,
            'val': val_loss,
            'val_edit': val_edit
        })
        
        # Save checkpoint
        store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f'checkpoint_{epoch:03d}.pt')
        )
        
        # Save config
        config_dict = {
            'dataset': args.dataset_name,
            'num_classes': len(classes),
            'visual_arch': visual_arch,
            'skeleton_arch': skeleton_arch,
            'temporal_arch': temporal_arch,
            'clip_len': clip_len,
            'batch_size': args.batch_size,
            'crop_dim': crop_dim,
            'window': window,
            'stage': 3,
            'stride': 2,
            'num_epochs': args.num_epochs,
            'warm_up_epochs': warm_up_epochs,
            'learning_rate': args.learning_rate,
            'start_val_epoch': args.num_epochs - 10,
            'gpu_parallel': False,
            'dilate_len': 0,
            'num_samples': -1
        }
        # Create a simple args-like object
        config_args = type('args', (), config_dict)()
        store_config(os.path.join(args.save_dir, 'config.json'), config_args, args.num_epochs, classes)
    
    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'Best epoch: {best_epoch} (Edit score: {best_edit_score:.4f})')
    print(f'Checkpoints saved to: {args.save_dir}')
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
