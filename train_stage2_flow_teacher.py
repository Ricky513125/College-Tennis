#!/usr/bin/env python3
"""
Train Stage 2 with Flow as Teacher (Ablation Study).

This is an ablation experiment where Flow is used as the teacher network,
and RGB and Skeleton learn from Flow features (instead of the original
MD-FED where Skeleton is the teacher).

Original MD-FED Stage 2:
- Teacher: Skeleton
- Students: RGB, Flow
- Loss: MSE(RGB_feat, Skeleton_feat) + MSE(Flow_feat, Skeleton_feat)

This ablation (Flow as Teacher):
- Teacher: Flow
- Students: RGB, Skeleton
- Loss: MSE(RGB_feat, Flow_feat) + MSE(Skeleton_feat, Flow_feat)
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import math
from contextlib import nullcontext

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

# Import MD-FED modules
import importlib.util
train_md_fed_path = os.path.join(md_fed_dir, 'train_MD-FED.py')
spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
train_MD_FED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_MD_FED)

MD_FED = train_MD_FED.MD_FED
get_best_epoch_and_history = train_MD_FED.get_best_epoch_and_history

from dataset.input_process import ActionSeqDataset
from util.dataset import load_classes
from util.io import load_json, store_json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR


class MD_FED_Flow_Teacher(MD_FED):
    """
    Modified MD-FED with Flow as teacher network.
    In Stage 2, RGB and Skeleton learn from Flow features.
    """
    
    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5):
        """
        Modified epoch function for Flow as teacher distillation.
        """
        if optimizer is None:
            self._model.eval()
            mode = "Validation"
        else:
            optimizer.zero_grad()
            self._model.train()
            mode = "Training"

        epoch_loss = 0.
        
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader, desc=mode)):
                frame = loader.dataset.load_frame_gpu(batch, self._device)
                flow = loader.dataset.load_flow_gpu(batch, self._device)
                skeleton = loader.dataset.load_skeleton_gpu(batch, self._device)

                with torch.amp.autocast('cuda'):
                    loss = 0.

                    # stage 2: multimodal distillation with Flow as teacher
                    if self._stage == 2:
                        # Get features from all modalities
                        _, _, rgb_feat, flow_feat, sk_feat = self._model(frame, flow, skeleton)
                        
                        # Flow as teacher: RGB and Skeleton learn from Flow
                        # L2 loss: students should match teacher
                        rgb2flow_loss = F.mse_loss(rgb_feat, flow_feat)
                        sk2flow_loss = F.mse_loss(sk_feat, flow_feat)
                        
                        loss += rgb2flow_loss
                        loss += sk2flow_loss
                        
                        # Log losses for monitoring
                        if batch_idx == 0 and mode == "Training":
                            print(f"\n[Flow as Teacher] RGB→Flow loss: {rgb2flow_loss.item():.6f}, Skeleton→Flow loss: {sk2flow_loss.item():.6f}")

                if optimizer is not None:
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % acc_grad_iter == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if lr_scheduler is not None:
                            lr_scheduler.step()

                epoch_loss += loss.item()

        return epoch_loss / len(loader)


def prepare_stage2_data(manual_annotations_file, output_dir, dataset_name='ncaa-rally', train_ratio=0.8):
    """
    Prepare Stage 2 training data (same as original Stage 2).
    """
    print(f"Preparing Stage 2 data for Flow as Teacher ablation...")
    
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # Shuffle and split
    import random
    random.seed(42)
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]
    
    # Save train/val splits
    train_json = data_dir / 'train.json'
    val_json = data_dir / 'val.json'
    
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Train: {len(train_annotations)} rallies")
    print(f"Val: {len(val_annotations)} rallies")
    
    # Copy elements.txt from MD-FED
    elements_src = Path('MD-FED/data/f3set-tennis-sub/elements.txt')
    if elements_src.exists():
        elements_dst = data_dir / 'elements.txt'
        import shutil
        shutil.copy(elements_src, elements_dst)
        print(f"Copied elements.txt")
    
    return str(data_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Train Stage 2 with Flow as Teacher (Ablation Study)'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        default='./manual_annotations.json',
        help='Path to manual annotations JSON file'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Directory containing RGB frames'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        required=True,
        help='Directory containing optical flow files'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        required=True,
        help='Directory containing skeleton pkl files'
    )
    parser.add_argument(
        '--stage1_model_dir',
        type=str,
        default='./md_fed_outputs/stage1',
        help='Directory containing Stage 1 trained model (for skeleton initialization)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save Stage 2 checkpoints (e.g., ./md_fed_outputs/stage2_flow_teacher)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./md_fed_data',
        help='Directory to save prepared data'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Dataset name'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default='rny002_tsm',
        help='Visual architecture'
    )
    parser.add_argument(
        '--skeleton_arch',
        type=str,
        default='stgcn++',
        help='Skeleton architecture'
    )
    parser.add_argument(
        '--temporal_arch',
        type=str,
        default='gru',
        help='Temporal architecture'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='Clip length'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=2,
        help='Frame stride'
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=224,
        help='Crop dimension'
    )
    parser.add_argument(
        '--warm_up_epochs',
        type=int,
        default=3,
        help='Warm-up epochs'
    )
    parser.add_argument(
        '--acc_grad_iter',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage 2 Training: Flow as Teacher (Ablation Study)")
    print("=" * 80)
    print("⚠️  消融实验：Flow 作为教师网络，RGB 和 Skeleton 学习 Flow 特征")
    print("=" * 80)
    
    # Step 1: Prepare data
    print("\nStep 1: Preparing data...")
    data_dir = prepare_stage2_data(
        args.manual_annotations,
        args.data_dir,
        args.dataset_name
    )
    
    # Step 2: Load classes
    elements_file = os.path.join(data_dir, 'elements.txt')
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Step 3: Create datasets
    print("\nStep 2: Creating datasets...")
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    epoch_num_frames = 100000  # Large dataset for Stage 2
    dataset_len = epoch_num_frames // (args.clip_len * args.stride)
    
    dataset_kwargs = {
        'crop_dim': args.crop_dim,
        'stride': args.stride
    }
    
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, args.clip_len, dataset_len,
        is_eval=False, dilate_len=0, stage=2,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
        **dataset_kwargs
    )
    train_data.print_info()
    
    val_data = ActionSeqDataset(
        classes, val_json,
        args.frame_dir, args.clip_len, dataset_len // 4,
        is_eval=True, dilate_len=0, stage=2,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
        **dataset_kwargs
    )
    val_data.print_info()
    
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=args.batch_size,
        pin_memory=True, num_workers=4
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=4
    )
    
    # Step 4: Create model
    print("\nStep 3: Creating model...")
    model = MD_FED_Flow_Teacher(
        len(classes),
        args.visual_arch,
        args.skeleton_arch,
        args.temporal_arch,
        clip_len=args.clip_len,
        step=args.stride,
        window=5,
        stage=2,  # Stage 2 for distillation
        multi_gpu=False
    )
    
    # Step 5: Load Stage 1 checkpoint (for skeleton initialization)
    print("\nStep 4: Loading Stage 1 checkpoint...")
    if os.path.exists(args.stage1_model_dir):
        losses, best_epoch, best_criterion = get_best_epoch_and_history(
            args.stage1_model_dir, 'loss'
        )
        print(f'Loading from Stage 1 epoch {best_epoch}')
        
        stage1_checkpoint = torch.load(
            os.path.join(args.stage1_model_dir, f'checkpoint_{best_epoch:03d}.pt'),
            map_location='cuda'
        )
        model.load(stage1_checkpoint)
        print("✓ Stage 1 checkpoint loaded (skeleton weights initialized)")
    else:
        print("⚠️  Warning: Stage 1 checkpoint not found. Skeleton will be randomly initialized.")
    
    # Step 6: Setup training
    print("\nStep 5: Setting up training...")
    optimizer = torch.optim.Adam(model._model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    num_steps_per_epoch = len(train_loader)
    warm_up_steps = args.warm_up_epochs * num_steps_per_epoch
    cosine_steps = (args.num_epochs - args.warm_up_epochs) * num_steps_per_epoch
    
    lr_scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warm_up_steps),
        CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=args.learning_rate * 0.01)
    ])
    
    # Step 7: Save configuration
    print("\nStep 6: Saving configuration...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    config = {
        'dataset': args.dataset_name,
        'num_classes': len(classes),
        'visual_arch': args.visual_arch,
        'skeleton_arch': args.skeleton_arch,
        'temporal_arch': args.temporal_arch,
        'num_samples': -1,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'window': 5,
        'stage': 2,
        'stride': args.stride,
        'num_epochs': args.num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.num_epochs - 20,
        'gpu_parallel': False,
        'dilate_len': 0
    }
    
    config_path = os.path.join(args.save_dir, 'config.json')
    store_json(config_path, config, pretty=True)
    print(f"✓ Configuration saved to {config_path}")
    
    # Step 8: Training loop
    print("\nStep 7: Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = model.epoch(
            train_loader, optimizer=optimizer, scaler=scaler,
            lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = model.epoch(val_loader, optimizer=None)
        val_losses.append(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'[Epoch {epoch+1}/{args.num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model._model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{epoch:03d}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f'  ✅ New best val loss: {val_loss:.6f}')
        
        # Save history
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        history_path = os.path.join(args.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.save_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
