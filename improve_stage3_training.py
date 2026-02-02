#!/usr/bin/env python3
"""
改进的 Stage 3 训练脚本
支持从最佳 epoch 继续训练，并提供多种优化选项
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

# Import train_MD-FED.py using importlib
import importlib.util
train_md_fed_path = os.path.join(md_fed_dir, 'train_MD-FED.py')
spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
train_MD_FED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_MD_FED)

MD_FED = train_MD_FED.MD_FED
evaluate = train_MD_FED.evaluate
get_best_epoch_and_history = train_MD_FED.get_best_epoch_and_history
store_config = train_MD_FED.store_config
get_last_epoch = train_MD_FED.get_last_epoch

from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json, store_json
from torch.utils.data import DataLoader
import random


def load_training_state(save_dir, model, optimizer, scaler, lr_scheduler, resume=True):
    """从检查点加载训练状态"""
    if not os.path.exists(save_dir):
        return 0, [], None, 0
    
    epoch = get_last_epoch(save_dir)
    if epoch < 0:
        return 0, [], None, 0
    
    print(f'Loading from epoch {epoch}')
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{epoch:03d}.pt')
    if os.path.exists(checkpoint_path):
        model.load(torch.load(checkpoint_path))
    
    losses = []
    best_epoch = None
    best_edit_score = 0
    
    loss_file = os.path.join(save_dir, 'loss.json')
    if os.path.exists(loss_file):
        losses = load_json(loss_file)
        # 找到最佳 epoch
        best_entry = max(losses, key=lambda x: x.get('val_edit', 0))
        best_epoch = best_entry['epoch']
        best_edit_score = best_entry.get('val_edit', 0)
        print(f'Best epoch so far: {best_epoch} (Edit score: {best_edit_score:.4f})')
    
    if resume and epoch >= 0:
        optim_path = os.path.join(save_dir, f'optim_{epoch:03d}.pt')
        if os.path.exists(optim_path):
            opt_data = torch.load(optim_path)
            optimizer.load_state_dict(opt_data['optimizer_state_dict'])
            scaler.load_state_dict(opt_data['scaler_state_dict'])
            lr_scheduler.load_state_dict(opt_data['lr_state_dict'])
            print('✓ Optimizer state loaded')
    
    return epoch + 1, losses, best_epoch, best_edit_score


def main():
    parser = argparse.ArgumentParser(
        description='Improved Stage 3 training with resume and optimization options'
    )
    
    # 基本参数
    parser.add_argument(
        '--stage2_checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing Stage 2 checkpoints'
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
        help='Directory containing optical flow files'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save Stage 3 checkpoints'
    )
    
    # 训练参数
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--resume_from_best',
        action='store_true',
        help='Resume training from best epoch (instead of last epoch)'
    )
    parser.add_argument(
        '--resume_from_epoch',
        type=int,
        default=None,
        help='Resume training from specific epoch'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of additional training epochs (default: 100)'
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
        default=None,
        help='Learning rate (default: auto from previous or 0.00005)'
    )
    parser.add_argument(
        '--reduce_lr',
        action='store_true',
        help='Reduce learning rate by 10x for fine-tuning'
    )
    
    # 评估参数
    parser.add_argument(
        '--eval_frequency',
        type=int,
        default=10,
        help='Evaluate every N epochs (default: 10)'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=50,
        help='Early stop patience (epochs without improvement, default: 50)'
    )
    
    # 数据参数
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Name of the dataset'
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
        help='Ratio of data to use for training'
    )
    
    # 其他参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("Improved Stage 3 Training")
    print(f"{'='*60}\n")
    
    # 检查是否已有训练结果
    existing_training = os.path.exists(args.save_dir) and os.path.exists(
        os.path.join(args.save_dir, 'loss.json')
    )
    
    if existing_training:
        print(f"\n{'='*60}")
        print(f"Found existing training in: {args.save_dir}")
        print(f"{'='*60}")
        
        # 读取训练历史
        loss_file = os.path.join(args.save_dir, 'loss.json')
        if os.path.exists(loss_file):
            losses = load_json(loss_file)
            if losses:
                last_epoch = max(e['epoch'] for e in losses)
                best_entry = max(losses, key=lambda x: x.get('val_edit', 0))
                best_epoch = best_entry['epoch']
                best_edit = best_entry.get('val_edit', 0)
                
                print(f"\n📊 Training History:")
                print(f"   Last epoch: {last_epoch}")
                print(f"   Best epoch: {best_epoch}")
                print(f"   Best Edit score: {best_edit:.4f}")
                print(f"   Total epochs trained: {len(losses)}")
        
        # 检查可用的检查点
        checkpoint_files = [f for f in os.listdir(args.save_dir) if f.startswith('checkpoint_')]
        if checkpoint_files:
            epochs = sorted([int(f.replace('checkpoint_', '').replace('.pt', '')) for f in checkpoint_files])
            print(f"\n📁 Available checkpoints:")
            print(f"   Epochs: {epochs[0]} to {epochs[-1]} (total: {len(epochs)})")
            if best_epoch in epochs:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{best_epoch:03d}.pt')
                print(f"   ✓ Best checkpoint exists: {checkpoint_path}")
        
        # 决定如何继续
        if args.resume_from_epoch is not None:
            print(f"\n🔄 Will resume from specified epoch: {args.resume_from_epoch}")
        elif args.resume_from_best:
            print(f"\n🔄 Will resume from BEST epoch: {best_epoch} (Edit score: {best_edit:.4f})")
        elif args.resume:
            print(f"\n🔄 Will resume from LAST epoch: {last_epoch}")
        else:
            print(f"\n⚠️  WARNING: Existing training found but no resume option specified!")
            print(f"   Options:")
            print(f"     --resume              : Continue from last epoch ({last_epoch})")
            print(f"     --resume_from_best    : Continue from best epoch ({best_epoch}, Edit: {best_edit:.4f})")
            print(f"     --resume_from_epoch N : Continue from specific epoch")
            print(f"   Or use a different --save_dir to start fresh")
            response = input("\nContinue from best epoch? (y/n): ")
            if response.lower() == 'y':
                args.resume_from_best = True
                print(f"✓ Will resume from best epoch {best_epoch}")
            else:
                print("Exiting. Please specify --resume, --resume_from_best, or use different --save_dir")
                return
    else:
        print(f"\n{'='*60}")
        print(f"Starting NEW training in: {args.save_dir}")
        print(f"{'='*60}")
    
    # 准备数据（如果不存在）
    from few_shot_learning_stage3 import prepare_few_shot_data, convert_manual_annotations_to_md_fed_format
    
    data_dir = prepare_few_shot_data(
        args.manual_annotations,
        args.data_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio
    )
    
    # 加载配置
    if existing_training:
        config_file = os.path.join(args.save_dir, 'config.json')
        if os.path.exists(config_file):
            config = load_json(config_file)
            print(f"\nLoading config from previous training...")
            visual_arch = config.get('visual_arch', 'rny002_tsm')
            skeleton_arch = config.get('skeleton_arch', 'stgcn++')
            temporal_arch = config.get('temporal_arch', 'gru')
            clip_len = config.get('clip_len', 96)
            crop_dim = config.get('crop_dim', 224)
            window = config.get('window', 5)
            
            # 使用之前的学习率或新的
            if args.learning_rate is None:
                args.learning_rate = config.get('learning_rate', 0.0001)
                if args.reduce_lr:
                    args.learning_rate *= 0.1
                    print(f"Reduced learning rate to: {args.learning_rate}")
        else:
            # 从 Stage 2 加载配置
            stage2_config_file = os.path.join(args.stage2_checkpoint_dir, 'config.json')
            if os.path.exists(stage2_config_file):
                config = load_json(stage2_config_file)
                visual_arch = config.get('visual_arch', 'rny002_tsm')
                skeleton_arch = config.get('skeleton_arch', 'stgcn++')
                temporal_arch = config.get('temporal_arch', 'gru')
                clip_len = config.get('clip_len', 96)
                crop_dim = config.get('crop_dim', 224)
                window = config.get('window', 5)
                args.learning_rate = args.learning_rate or 0.00005
            else:
                raise FileNotFoundError("Cannot find config file")
    else:
        # 从 Stage 2 加载配置
        stage2_config_file = os.path.join(args.stage2_checkpoint_dir, 'config.json')
        if not os.path.exists(stage2_config_file):
            raise FileNotFoundError(f"Stage 2 config not found: {stage2_config_file}")
        
        config = load_json(stage2_config_file)
        visual_arch = config.get('visual_arch', 'rny002_tsm')
        skeleton_arch = config.get('skeleton_arch', 'stgcn++')
        temporal_arch = config.get('temporal_arch', 'gru')
        clip_len = config.get('clip_len', 96)
        crop_dim = config.get('crop_dim', 224)
        window = config.get('window', 5)
        args.learning_rate = args.learning_rate or 0.0001
    
    print(f"\nConfiguration:")
    print(f"  Visual arch: {visual_arch}")
    print(f"  Temporal arch: {temporal_arch}")
    print(f"  Clip len: {clip_len}")
    print(f"  Crop dim: {crop_dim}")
    print(f"  Window: {window}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # 加载类别
    elements_file = os.path.join(data_dir, 'elements.txt')
    if not os.path.exists(elements_file):
        raise FileNotFoundError(f"Elements file not found: {elements_file}")
    
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # 创建数据集
    print(f"\nCreating datasets...")
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    epoch_num_frames = 10000
    dataset_len = epoch_num_frames // (clip_len * 2)
    
    dataset_kwargs = {
        'crop_dim': crop_dim,
        'stride': 2
    }
    
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, clip_len, dataset_len,
        is_eval=False, dilate_len=0, stage=3,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=None,
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
    
    # 创建模型
    print(f"\nCreating model...")
    model = MD_FED(
        len(classes),
        visual_arch,
        skeleton_arch,
        temporal_arch,
        clip_len=clip_len,
        step=2,
        window=window,
        stage=3,
        multi_gpu=False
    )
    
    # 加载模型 - 明确显示加载路径
    print(f"\n{'='*60}")
    print("Loading Model")
    print(f"{'='*60}")
    
    if existing_training and (args.resume or args.resume_from_best or args.resume_from_epoch):
        # 从已有训练继续
        if args.resume_from_epoch is not None:
            # 从指定 epoch 加载
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{args.resume_from_epoch:03d}.pt')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
            
            print(f"📂 Loading checkpoint from:")
            print(f"   Path: {os.path.abspath(checkpoint_path)}")
            print(f"   Epoch: {args.resume_from_epoch}")
            
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load(checkpoint)
            start_epoch = args.resume_from_epoch + 1
            
            # 加载训练历史
            loss_file = os.path.join(args.save_dir, 'loss.json')
            if os.path.exists(loss_file):
                losses = load_json(loss_file)
                best_entry = max(losses, key=lambda x: x.get('val_edit', 0))
                best_epoch = best_entry['epoch']
                best_edit_score = best_entry.get('val_edit', 0)
            else:
                losses = []
                best_epoch = None
                best_edit_score = 0
            
            print(f"✓ Model loaded successfully")
            print(f"   Will continue from epoch {start_epoch}")
            
        else:
            # 从最佳或最后 epoch 加载
            losses, best_epoch_hist, best_edit_score_hist = get_best_epoch_and_history(
                args.save_dir, 'edit'
            )
            
            if args.resume_from_best and best_epoch_hist is not None:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{best_epoch_hist:03d}.pt')
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"❌ Best checkpoint not found: {checkpoint_path}")
                
                print(f"📂 Loading BEST checkpoint from:")
                print(f"   Path: {os.path.abspath(checkpoint_path)}")
                print(f"   Epoch: {best_epoch_hist}")
                print(f"   Edit score: {best_edit_score_hist:.4f}")
                
                checkpoint = torch.load(checkpoint_path, map_location=args.device)
                model.load(checkpoint)
                start_epoch = best_epoch_hist + 1
                best_epoch = best_epoch_hist
                best_edit_score = best_edit_score_hist
                
                print(f"✓ Model loaded successfully")
                print(f"   Will continue from epoch {start_epoch}")
                
            else:
                last_epoch = get_last_epoch(args.save_dir)
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{last_epoch:03d}.pt')
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"❌ Last checkpoint not found: {checkpoint_path}")
                
                print(f"📂 Loading LAST checkpoint from:")
                print(f"   Path: {os.path.abspath(checkpoint_path)}")
                print(f"   Epoch: {last_epoch}")
                
                checkpoint = torch.load(checkpoint_path, map_location=args.device)
                model.load(checkpoint)
                start_epoch = last_epoch + 1
                
                # 加载训练历史
                loss_file = os.path.join(args.save_dir, 'loss.json')
                if os.path.exists(loss_file):
                    losses = load_json(loss_file)
                    best_entry = max(losses, key=lambda x: x.get('val_edit', 0))
                    best_epoch = best_entry['epoch']
                    best_edit_score = best_entry.get('val_edit', 0)
                else:
                    losses = []
                    best_epoch = None
                    best_edit_score = 0
                
                print(f"✓ Model loaded successfully")
                print(f"   Will continue from epoch {start_epoch}")
    else:
        # 从 Stage 2 开始
        print(f"📂 Loading Stage 2 checkpoint from:")
        print(f"   Directory: {os.path.abspath(args.stage2_checkpoint_dir)}")
        
        losses_stage2, best_epoch_stage2, best_edit_stage2 = get_best_epoch_and_history(
            args.stage2_checkpoint_dir, 'edit'
        )
        
        stage2_checkpoint_path = os.path.join(args.stage2_checkpoint_dir, f'checkpoint_{best_epoch_stage2:03d}.pt')
        print(f"   Path: {os.path.abspath(stage2_checkpoint_path)}")
        print(f"   Epoch: {best_epoch_stage2}")
        print(f"   Edit score: {best_edit_stage2:.4f}")
        
        if not os.path.exists(stage2_checkpoint_path):
            raise FileNotFoundError(f"❌ Stage 2 checkpoint not found: {stage2_checkpoint_path}")
        
        stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location=args.device)
        model.load(stage2_checkpoint)
        start_epoch = 0
        losses = []
        best_epoch = None
        best_edit_score = 0
        
        print(f"✓ Stage 2 model loaded successfully")
        print(f"   Starting NEW training from epoch 0")
    
    # 设置优化器
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
    
    # 如果继续训练，加载优化器状态
    if existing_training and (args.resume or args.resume_from_best or args.resume_from_epoch):
        if args.resume_from_epoch is not None:
            epoch_to_load = args.resume_from_epoch
        elif args.resume_from_best and best_epoch is not None:
            epoch_to_load = best_epoch
        else:
            epoch_to_load = get_last_epoch(args.save_dir)
        
        optim_path = os.path.join(args.save_dir, f'optim_{epoch_to_load:03d}.pt')
        if os.path.exists(optim_path):
            print(f"\n📂 Loading optimizer state from:")
            print(f"   Path: {os.path.abspath(optim_path)}")
            opt_data = torch.load(optim_path)
            optimizer.load_state_dict(opt_data['optimizer_state_dict'])
            scaler.load_state_dict(opt_data['scaler_state_dict'])
            lr_scheduler.load_state_dict(opt_data['lr_state_dict'])
            print("✓ Optimizer state loaded successfully")
        else:
            print(f"⚠️  Optimizer state not found at: {optim_path}")
            print("   Will start with new optimizer state")
    
    # 训练循环
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"  Start epoch: {start_epoch}")
    print(f"  Additional epochs: {args.num_epochs}")
    print(f"  Total epochs: {start_epoch + args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Eval frequency: every {args.eval_frequency} epochs")
    print(f"  Early stop patience: {args.early_stop_patience} epochs")
    if best_epoch is not None:
        print(f"  Previous best: epoch {best_epoch} (Edit score: {best_edit_score:.4f})")
    print(f"  Save directory: {os.path.abspath(args.save_dir)}")
    print(f"{'='*60}\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    no_improve_count = 0
    
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=1)
        val_loss = model.epoch(val_loader, acc_grad_iter=1)
        
        print(f'[Epoch {epoch}] Train loss: {train_loss:.5f} Val loss: {val_loss:.5f}')
        
        # 评估
        val_edit = 0
        if epoch % args.eval_frequency == 0 or epoch >= start_epoch + args.num_epochs - 10:
            val_edit = evaluate(
                model, val_data_frames, classes,
                delta=10, window=window, dataset_name=args.dataset_name
            )
            if val_edit > best_edit_score:
                best_edit_score = val_edit
                best_epoch = epoch
                no_improve_count = 0
                print(f'✓ New best epoch! Edit score: {val_edit:.4f}')
            else:
                no_improve_count += args.eval_frequency
                print(f'  Edit score: {val_edit:.4f} (best: {best_edit_score:.4f} @ epoch {best_epoch})')
        
        losses.append({
            'epoch': epoch,
            'train': train_loss,
            'val': val_loss,
            'val_edit': val_edit
        })
        
        # 保存检查点
        store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f'checkpoint_{epoch:03d}.pt')
        )
        torch.save(
            {
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'lr_state_dict': lr_scheduler.state_dict()
            },
            os.path.join(args.save_dir, f'optim_{epoch:03d}.pt')
        )
        
        # 早停
        if no_improve_count >= args.early_stop_patience:
            print(f'\n⚠️  Early stopping: No improvement for {args.early_stop_patience} epochs')
            break
    
    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f"{'="*60}")
    print(f'\n📊 Final Results:')
    print(f'   Best epoch: {best_epoch}')
    print(f'   Best Edit score: {best_edit_score:.4f}')
    
    if best_epoch is not None:
        best_checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{best_epoch:03d}.pt')
        print(f'\n📂 Best Model Checkpoint:')
        print(f'   Path: {os.path.abspath(best_checkpoint_path)}')
        print(f'   Epoch: {best_epoch}')
        print(f'   Edit score: {best_edit_score:.4f}')
        
        # 验证文件存在
        if os.path.exists(best_checkpoint_path):
            file_size = os.path.getsize(best_checkpoint_path) / (1024 * 1024)  # MB
            print(f'   Size: {file_size:.2f} MB')
            print(f'   ✓ Checkpoint file exists')
        else:
            print(f'   ⚠️  WARNING: Checkpoint file not found!')
    
    print(f'\n📁 All checkpoints saved to:')
    print(f'   {os.path.abspath(args.save_dir)}')
    print(f'\n💡 To use this model for testing:')
    print(f'   python test_stage2_on_manual_data.py \\')
    print(f'       --checkpoint_dir {args.save_dir} \\')
    print(f'       --epoch {best_epoch} \\')
    print(f'       --manual_annotations manual_annotations.json \\')
    print(f'       --frame_dir /path/to/frames \\')
    print(f'       --flow_dir /path/to/flow')
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
