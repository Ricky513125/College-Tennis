#!/usr/bin/env python3
"""
Train I3D model for comparison with MD-FED Stage 3.

This script trains an I3D (Inflated 3D ConvNet) model on the manual_annotations.json
dataset and evaluates its performance. The goal is to compare I3D with the MD-FED
Stage 3 model and VTN model.

Usage:
    python train_i3d_comparison.py \\
        --manual_annotations manual_annotations.json \\
        --frame_dir /path/to/frames \\
        --save_dir ./i3d_outputs \\
        --crop_dim 224 \\
        --clip_len 96 \\
        --batch_size 4 \\
        --num_epochs 500
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Add MD-FED to path
sys.path.insert(0, str(Path(__file__).parent / 'MD-FED'))

# Import necessary modules
from model.pytorch_i3d import InceptionI3d
from util.dataset import load_classes
from util.io import load_json, store_json
from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import random
import numpy as np
from tqdm import tqdm


class I3D_MD_FED(nn.Module):
    """
    使用 I3D 作为 backbone 的 MD-FED 模型
    """
    def __init__(self, num_classes, clip_len, pretrained=True, dropout=0.5):
        super().__init__()
        
        self._num_classes = num_classes
        self._clip_len = clip_len
        
        # I3D backbone (extract features)
        # final_endpoint='Mixed_4f' 提取中间特征，避免过拟合
        self.i3d = InceptionI3d(
            num_classes=400,  # Kinetics 预训练
            spatial_squeeze=False,
            final_endpoint='Mixed_4f',  # 在 Mixed_4f 层停止
            in_channels=3,
            dropout_keep_prob=dropout
        )
        
        # I3D Mixed_4f 输出: [batch, 832, T/2, 7, 7] (对于 224x224 输入)
        # 832 = 256+160+320+32+128+128
        i3d_output_channels = 832
        
        # Temporal pooling and classification
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 只池化空间维度
        
        # Temporal head for sequence modeling
        self.temporal_head = nn.GRU(
            i3d_output_channels, 368, num_layers=1, batch_first=True, bidirectional=False
        )
        
        # Prediction heads
        self.coarse_fc = nn.Linear(368, 2)  # Event/Non-event
        self.fine_fc = nn.Linear(368, num_classes)  # Fine-grained classes
        
    def forward(self, frames, flow=None, skeleton=None):
        """
        Args:
            frames: [batch_size, clip_len, C, H, W]
            flow: Not used in I3D (only RGB)
            skeleton: Not used in I3D
        """
        batch_size, clip_len, c, h, w = frames.shape
        
        # I3D expects [batch, channel, frames, height, width]
        x = frames.permute(0, 2, 1, 3, 4)  # [batch, C, T, H, W]
        
        # Extract features using I3D
        x = self.i3d(x)  # [batch, 832, T/2, 7, 7]
        
        # Spatial pooling
        x = self.avg_pool(x)  # [batch, 832, T/2, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch, 832, T/2]
        x = x.permute(0, 2, 1)  # [batch, T/2, 832]
        
        # Temporal modeling with GRU
        features, _ = self.temporal_head(x)  # [batch, T/2, 368]
        
        # Upsample to original temporal resolution
        # features: [batch, T/2, 368] -> [batch, T, 368]
        features = torch.nn.functional.interpolate(
            features.permute(0, 2, 1),  # [batch, 368, T/2]
            size=clip_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [batch, T, 368]
        
        # Predictions for each frame
        coarse_pred = self.coarse_fc(features)  # [batch, T, 2]
        fine_pred = self.fine_fc(features)  # [batch, T, num_classes]
        
        return coarse_pred, fine_pred
    
    def predict(self, frames, flow=None, skeleton=None):
        """
        Predict method for evaluation (compatible with MD-FED evaluation)
        Returns: (coarse_scores, fine_scores) as logits
        """
        coarse_pred, fine_pred = self.forward(frames, flow, skeleton)
        # Return logits (not softmax/sigmoid) for evaluation
        return None, coarse_pred, fine_pred
    
    def compute_loss(self, coarse_pred, fine_pred, coarse_label, fine_label):
        """
        计算损失
        """
        # Reshape predictions and labels
        batch_size, seq_len, _ = coarse_pred.shape
        
        coarse_pred = coarse_pred.reshape(-1, 2)
        fine_pred = fine_pred.reshape(-1, self._num_classes)
        coarse_label = coarse_label.reshape(-1)
        fine_label = fine_label.reshape(-1, self._num_classes)
        
        # Coarse loss (event detection)
        coarse_loss = nn.functional.cross_entropy(coarse_pred, coarse_label)
        
        # Fine loss (multi-label classification)
        fine_loss = nn.functional.binary_cross_entropy_with_logits(fine_pred, fine_label.float())
        
        # Total loss
        loss = coarse_loss + fine_loss
        
        return loss, coarse_loss, fine_loss


def prepare_i3d_data(args):
    """
    准备训练和验证数据（与 VTN 相同的分割策略）
    """
    print("Preparing I3D training data...")
    
    # Load annotations
    annotations = load_json(args.manual_annotations)
    
    # Split into train and val (same split as few_shot_learning_stage3.py)
    random.seed(42)
    indices = list(range(len(annotations)))
    random.shuffle(indices)
    
    split_point = int(len(annotations) * 0.8)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_annotations = [annotations[i] for i in train_indices]
    val_annotations = [annotations[i] for i in val_indices]
    
    print(f"Total annotations: {len(annotations)}")
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    
    # Save splits
    os.makedirs(args.save_dir, exist_ok=True)
    store_json(os.path.join(args.save_dir, 'train_annotations.json'), train_annotations)
    store_json(os.path.join(args.save_dir, 'val_annotations.json'), val_annotations)
    
    # Copy elements.txt
    import shutil
    if os.path.exists('elements.txt'):
        shutil.copy('elements.txt', os.path.join(args.save_dir, 'elements.txt'))
        print("Copied elements.txt")
    
    return train_annotations, val_annotations


def train_i3d(args):
    """
    Train I3D model
    """
    print("\n" + "="*80)
    print("Training I3D Model")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Prepare data
    print("\nStep 1: Preparing data...")
    train_annotations, val_annotations = prepare_i3d_data(args)
    
    # Load classes
    classes = load_classes('elements.txt')
    print(f"Loaded {len(classes)} classes")
    
    # Create model
    print("\nStep 2: Creating I3D model...")
    use_pretrained = not args.no_pretrained
    if use_pretrained:
        print("✅ 将使用 Kinetics 预训练权重（如果可用）")
    else:
        print("⚠️  将从随机初始化开始训练（不使用预训练）")
    
    model = I3D_MD_FED(
        num_classes=len(classes),
        clip_len=args.clip_len,
        pretrained=use_pretrained,
        dropout=args.dropout
    ).cuda()
    
    # Load pretrained weights if available
    if use_pretrained and args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"📂 Loading pretrained I3D weights from: {args.pretrained_path}")
        try:
            pretrained_dict = torch.load(args.pretrained_path)
            model_dict = model.i3d.state_dict()
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.i3d.load_state_dict(model_dict, strict=False)
            print(f"✅ Successfully loaded {len(pretrained_dict)} layers from pretrained weights")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained weights: {e}")
            print("⚠️  Will train from scratch")
    
    print(f"Model configuration:")
    print(f"  Backbone: I3D (Inception-v1)")
    print(f"  Clip len: {args.clip_len}")
    print(f"  Image size: {args.crop_dim}×{args.crop_dim}")
    print(f"  Dropout: {args.dropout}")
    
    # Create datasets
    print("\nStep 3: Creating datasets...")
    train_data = ActionSeqDataset(
        classes=classes,
        label_file=os.path.join(args.save_dir, 'train_annotations.json'),
        frame_dir=args.frame_dir,
        clip_len=args.clip_len,
        dataset_len=args.dataset_len,
        flow_dir=args.flow_dir,
        crop_dim=args.crop_dim,
        stride=args.stride,
        is_eval=False,  # 训练时使用随机裁剪
        stage=3
    )
    
    val_data = ActionSeqDataset(
        classes=classes,
        label_file=os.path.join(args.save_dir, 'val_annotations.json'),
        frame_dir=args.frame_dir,
        clip_len=args.clip_len,
        dataset_len=args.dataset_len // 4,
        flow_dir=args.flow_dir,
        crop_dim=args.crop_dim,
        stride=args.stride,
        is_eval=True,  # 验证时使用居中裁剪
        stage=3
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Setup training
    print("\nStep 4: Setup training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs * len(train_loader)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.num_epochs - args.warmup_epochs) * len(train_loader),
        eta_min=args.learning_rate * 0.01
    )
    lr_scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    print("\nStep 5: Training...")
    best_val_loss = float('inf')
    losses = []
    
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', leave=False)
        for batch in train_pbar:
            frames = batch['frame'].cuda()
            coarse_label = batch['coarse_label'].cuda()
            fine_label = batch['fine_label'].cuda()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                coarse_pred, fine_pred = model(frames)
                loss, coarse_loss, fine_loss = model.compute_loss(
                    coarse_pred, fine_pred, coarse_label, fine_label
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            train_loss += loss.item()
            # Update progress bar with current batch loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                frames = batch['frame'].cuda()
                coarse_label = batch['coarse_label'].cuda()
                fine_label = batch['fine_label'].cuda()
                
                coarse_pred, fine_pred = model(frames)
                loss, _, _ = model.compute_loss(
                    coarse_pred, fine_pred, coarse_label, fine_label
                )
                
                val_loss += loss.item()
                # Update progress bar with current batch loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'[Epoch {epoch+1}/{args.num_epochs}] Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {current_lr:.2e}')
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'  ✅ New best val loss: {val_loss:.5f} → Saving checkpoint...')
            # Save best model separately
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, 'best_model.pt')
            )
        
        losses.append({
            'epoch': epoch,
            'train': train_loss,
            'val': val_loss
        })
        
        store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f'checkpoint_{epoch:03d}.pt')
        )
    
    print(f'\n{"="*80}')
    print(f'Training Complete!')
    print(f'Best validation loss: {best_val_loss:.5f}')
    print(f'Models saved to: {args.save_dir}')
    print(f'{"="*80}\n')
    
    # Evaluate best model if requested
    if args.evaluate_after_training:
        print("\n" + "="*80)
        print("Evaluating Best Model")
        print("="*80)
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
        model.eval()
        
        # Import evaluation function from MD-FED
        sys.path.insert(0, str(Path(__file__).parent / 'MD-FED'))
        from train_MD_FED import evaluate as md_fed_evaluate
        
        # Create evaluation dataset (use validation dataset)
        eval_dataset = ActionSeqVideoDataset(
            classes=classes,
            label_file=os.path.join(args.save_dir, 'val_annotations.json'),
            frame_dir=args.frame_dir,
            clip_len=args.clip_len,
            crop_dim=args.crop_dim,
            stride=args.stride,
            is_eval=True,
            stage=3
        )
        
        # Evaluate using MD-FED's evaluation function
        print("\nRunning evaluation (this may take a while)...")
        edit_score = md_fed_evaluate(
            model, eval_dataset, classes, 
            delta=1, window=5, 
            dataset_name='ncaa-rally', 
            device='cuda'
        )
        
        print(f"\n✓ Evaluation complete!")
        print(f"  Edit Score: {edit_score:.4f}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train I3D for comparison with MD-FED')
    
    # Data parameters
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
        '--flow_dir',
        type=str,
        default=None,
        help='Path to optical flow directory (not used by I3D)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./i3d_outputs',
        help='Directory to save model checkpoints and logs'
    )
    
    # Model parameters
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=224,
        help='Crop dimension for images (default: 224)'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='Number of frames per clip (default: 96)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=2,
        help='Frame sampling stride (default: 2)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training (default: 4)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
        help='Number of training epochs (default: 500)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=10,
        help='Number of warmup epochs (default: 10)'
    )
    parser.add_argument(
        '--dataset_len',
        type=int,
        default=1000,
        help='Number of clips per epoch (default: 1000)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Pretrained weights
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='不使用预训练权重（从头训练）'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default=None,
        help='Path to pretrained I3D weights (Kinetics pretrained)'
    )
    parser.add_argument(
        '--evaluate_after_training',
        action='store_true',
        help='Evaluate the best model after training (computes Mean F1 (LCL), Mean F1 (event), Mean F1 (element), Edit Score)'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_i3d(args)


if __name__ == '__main__':
    main()
