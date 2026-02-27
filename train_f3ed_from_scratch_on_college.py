#!/usr/bin/env python3
"""
实验2：F3ED不用F3Set训练，只在少量校园网球训练的效果（从头训练）

这个实验展示从头训练 vs 迁移学习的对比：
- 不进行F3Set预训练
- 只在校园网球数据集上从头训练
- 使用与I3D/TSM等对比模型相同的数据划分和训练策略

Usage:
    python train_f3ed_from_scratch_on_college.py \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --save_dir ./f3ed_from_scratch_outputs \
        --batch_size 4 \
        --num_epochs 100 \
        --learning_rate 0.001
"""

import os
import sys
import json
import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm

# Add F3Set to path
f3set_dir = os.path.join(os.path.dirname(__file__), 'F3Set')
if os.path.exists(f3set_dir):
    sys.path.insert(0, f3set_dir)

# Import F3ED model and utilities
from train_f3set_f3ed import F3Set, evaluate, get_lr_scheduler, store_config, EPOCH_NUM_FRAMES
from dataset.frame_process import ActionSeqDataset, ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json, store_json
from model.common import step
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import shutil


def prepare_college_tennis_data(manual_annotations_file, output_dir, dataset_name='college-tennis', train_ratio=0.8):
    """
    准备校园网球训练数据（与I3D/TSM等对比模型相同的数据划分）
    """
    print(f"Preparing college tennis training data...")
    
    # Create data directory
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manual annotations
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # Shuffle and split (same seed as other comparison models)
    random.seed(42)
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]
    
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    
    # Save train.json and val.json
    train_file = data_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved train data to {train_file}")
    
    val_file = data_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved val data to {val_file}")
    
    # Copy elements.txt from various possible locations
    elements_src = None
    possible_sources = [
        os.path.join('F3Set', 'data', 'f3set-tennis', 'elements.txt'),
        os.path.join('MD-FED', 'data', 'f3set-tennis-sub', 'elements.txt'),
        os.path.join('data', 'f3set-tennis', 'elements.txt'),
        'elements.txt',
    ]
    
    for src in possible_sources:
        if os.path.exists(src):
            elements_src = src
            break
    
    if elements_src:
        elements_dst = data_dir / 'elements.txt'
        shutil.copy(elements_src, elements_dst)
        print(f"Copied elements.txt from {elements_src} to {elements_dst}")
    else:
        print("⚠️  Warning: elements.txt not found in any expected location")
    
    return str(data_dir)


def get_datasets(data_dir, frame_dir, clip_len, crop_dim, stride, dataset_len=None):
    """
    Create train and validation datasets
    
    Args:
        dataset_len: Number of clips per epoch. If None, will be calculated based on EPOCH_NUM_FRAMES
    """
    elements_file = os.path.join(data_dir, 'elements.txt')
    classes = load_classes(elements_file)
    
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    # Calculate dataset_len if not provided
    if dataset_len is None:
        dataset_len = EPOCH_NUM_FRAMES // (clip_len * stride)
    
    print(f'Dataset size: {dataset_len} clips per epoch')
    
    train_data = ActionSeqDataset(
        classes, train_json, frame_dir, clip_len, dataset_len,
        is_eval=False,  # Training mode
        crop_dim=crop_dim, stride=stride
    )
    
    # Validation dataset uses smaller dataset_len
    val_dataset_len = dataset_len // 4
    val_data = ActionSeqDataset(
        classes, val_json, frame_dir, clip_len, val_dataset_len,
        is_eval=True,  # Evaluation mode
        crop_dim=crop_dim, stride=stride
    )
    
    # For evaluation during training
    val_data_frames = ActionSeqVideoDataset(
        classes, val_json, frame_dir, clip_len,
        overlap_len=clip_len // 2, crop_dim=crop_dim, stride=stride
    )
    
    return classes, train_data, val_data, val_data_frames


def train_f3ed_from_scratch(args):
    """
    在校园网球数据集上从头训练F3ED模型
    """
    print("="*70)
    print("实验2：F3ED不用F3Set训练，只在少量校园网球训练的效果（从头训练）")
    print("="*70)
    
    # Step 1: Prepare data
    print("\nStep 1: Preparing data...")
    data_dir = prepare_college_tennis_data(
        args.manual_annotations,
        args.save_dir,
        dataset_name='college-tennis',
        train_ratio=0.8
    )
    
    # Step 2: Create datasets
    print("\nStep 2: Creating datasets...")
    classes, train_data, val_data, val_data_frames = get_datasets(
        data_dir,
        args.frame_dir,
        args.clip_len,
        args.crop_dim,
        args.stride
    )
    
    train_data.print_info()
    val_data.print_info()
    
    # Step 3: Create model (from scratch, no pretraining)
    print("\nStep 3: Creating F3ED model (from scratch)...")
    model = F3Set(
        len(classes),
        args.feature_arch,
        args.temporal_arch,
        clip_len=args.clip_len,
        step=args.stride,
        window=args.window,
        use_ctx=args.use_ctx,
        multi_gpu=args.gpu_parallel
    )
    
    # Note: Model will use ImageNet pretrained backbone, but NOT F3Set pretrained weights
    print("  Model initialized from scratch (ImageNet pretrained backbone only)")
    print(f"  Feature arch: {args.feature_arch}")
    print(f"  Temporal arch: {args.temporal_arch}")
    print(f"  Use CTX: {args.use_ctx}")
    
    # Step 4: Setup optimizer and scheduler
    print("\nStep 4: Setting up optimizer and scheduler...")
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    
    def worker_init_fn(id):
        random.seed(id + 42)
    
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data,
        shuffle=False,
        batch_size=loader_batch_size,
        pin_memory=True,
        num_workers=args.num_workers or 4,
        prefetch_factor=1,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=loader_batch_size,
        pin_memory=True,
        num_workers=args.num_workers or 4,
        worker_init_fn=worker_init_fn
    )
    
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch
    )
    
    # Step 5: Training loop
    print("\nStep 5: Starting training...")
    print("="*70)
    
    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'edit' else float('inf')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Store config (create a custom config dict since we don't have args.dataset)
    config = {
        'dataset': 'college-tennis',  # Custom dataset name
        'num_classes': len(classes),
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'use_ctx': args.use_ctx,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'window': args.window,
        'stride': args.stride,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES
    }
    store_json(os.path.join(args.save_dir, 'config.json'), config, pretty=True)
    
    for epoch in range(num_epochs):
        # Train
        train_loss = model.epoch(
            train_loader,
            optimizer,
            scaler,
            lr_scheduler=lr_scheduler,
            acc_grad_iter=args.acc_grad_iter,
            epoch=epoch
        )
        
        # Validate
        val_loss = model.epoch(
            val_loader,
            acc_grad_iter=args.acc_grad_iter,
            epoch=epoch
        )
        
        print(f'[Epoch {epoch+1}/{num_epochs}] Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}')
        
        # Evaluate with edit score if needed
        val_edit = 0
        if args.criterion == 'edit':
            if epoch >= args.start_val_epoch:
                val_edit = evaluate(model, val_data_frames, classes, window=args.window)
                print(f'  Edit score: {val_edit:.4f}')
                if val_edit > best_criterion:
                    best_criterion = val_edit
                    best_epoch = epoch
                    print('  ✓ New best epoch!')
        elif args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('  ✓ New best epoch!')
        
        # Save checkpoint
        losses.append({
            'epoch': epoch,
            'train': train_loss,
            'val': val_loss,
            'val_edit': val_edit
        })
        
        store_json(
            os.path.join(args.save_dir, 'loss.json'),
            losses,
            pretty=True
        )
        
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))
        )
        
        # Save optimizer state
        torch.save(
            {
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'lr_state_dict': lr_scheduler.state_dict()
            },
            os.path.join(args.save_dir, 'optim_{:03d}.pt'.format(epoch))
        )
    
    print("\n" + "="*70)
    print(f"Training Complete! Best epoch: {best_epoch}")
    print("="*70)
    
    # Load best model and evaluate
    if best_epoch is not None:
        print(f"\nLoading best model from epoch {best_epoch}...")
        model.load(torch.load(
            os.path.join(args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))
        ))
        
        # Final evaluation on validation set
        print("\nFinal evaluation on validation set...")
        final_edit = evaluate(model, val_data_frames, classes, window=args.window)
        
        # Save final results
        final_results = {
            'experiment': 'F3ED trained from scratch on college tennis (no F3Set pretraining)',
            'best_epoch': best_epoch,
            'best_criterion': best_criterion,
            'final_edit_score': final_edit,
            'num_train_videos': len(train_data._src_data),
            'num_val_videos': len(val_data._src_data),
        }
        
        results_file = os.path.join(args.save_dir, 'final_results.json')
        store_json(results_file, final_results, pretty=True)
        print(f"\nFinal results saved to: {results_file}")
    
    print("\nThis experiment shows the performance when training from scratch:")
    print("  - F3ED model trained only on college tennis dataset")
    print("  - No F3Set pretraining")
    print("  - Compare with evaluate_f3ed_pretrained_on_college.py for ablation")


def main():
    parser = argparse.ArgumentParser(
        description='Train F3ED from scratch on college tennis dataset'
    )
    
    # Data arguments
    parser.add_argument(
        '--manual_annotations',
        type=str,
        required=True,
        help='Path to manual_annotations.json'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Path to frame directory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save checkpoints and results'
    )
    
    # Model arguments
    parser.add_argument(
        '-m', '--feature_arch',
        type=str,
        default='rny002_tsm',
        choices=['rny002', 'rny002_tsm', 'rny008', 'rny008_tsm', 'rn50', 'rn50_tsm', 'slowfast'],
        help='Feature extraction architecture'
    )
    parser.add_argument(
        '-t', '--temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru', 'mstcn', 'asformer', 'actionformer', 'gcn', 'tcn', 'fc'],
        help='Temporal architecture'
    )
    parser.add_argument(
        '-ctx', '--use_ctx',
        action='store_true',
        help='Use contextual module'
    )
    
    # Training arguments
    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--acc_grad_iter', type=int, default=1)
    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--start_val_epoch', type=int, default=30)
    parser.add_argument('--criterion', choices=['edit', 'loss'], default='edit')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    
    args = parser.parse_args()
    
    # Add evaluate_after_training flag for consistency
    args.evaluate_after_training = True
    
    train_f3ed_from_scratch(args)


if __name__ == '__main__':
    main()
