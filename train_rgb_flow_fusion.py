#!/usr/bin/env python3
"""
Train RGB + Flow Fusion model for ablation study.

This script trains a model that fuses RGB and Flow features together,
without using skeleton data. This is an ablation study to compare with
MD-FED (which uses RGB + Flow + Skeleton).

Usage:
    python train_rgb_flow_fusion.py \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --flow_dir /path/to/flows \
        --save_dir ./rgb_flow_fusion_outputs \
        --crop_dim 224 \
        --clip_len 96 \
        --batch_size 4 \
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
import timm
from model.shift import make_temporal_shift
from util.dataset import load_classes
from util.io import load_json, store_json
from dataset.input_process import ActionSeqDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import random
import numpy as np
from tqdm import tqdm

HIDDEN_DIM = 368


class RGB_Flow_Fusion_MD_FED(nn.Module):
    """
    融合 RGB 和 Flow 特征的模型（不使用 Skeleton）
    用于消融实验
    """
    def __init__(self, num_classes, clip_len, visual_arch='rny002_tsm', 
                 temporal_arch='gru', fusion_method='add', pretrained=True):
        super().__init__()
        
        self._num_classes = num_classes
        self._clip_len = clip_len
        self._fusion_method = fusion_method
        
        # RGB feature extractor
        if 'rny002' in visual_arch:
            rgb_feat = timm.create_model('regnety_002', pretrained=pretrained)
            rgb_feat_dim = rgb_feat.head.fc.in_features
            rgb_feat.head.fc = nn.Identity()
            rgb_feat_dim = 368
        elif 'rn50' in visual_arch:
            import torchvision.models as models
            rgb_feat = models.resnet50(pretrained=pretrained)
            rgb_feat_dim = rgb_feat.fc.in_features
            rgb_feat.fc = nn.Identity()
            rgb_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported visual_arch: {visual_arch}")
        
        # Flow feature extractor
        if 'rny002' in visual_arch:
            flow_feat = timm.create_model('regnety_002', pretrained=pretrained)
            flow_feat_dim = flow_feat.head.fc.in_features
            flow_feat.head.fc = nn.Identity()
            # Modify first layer to accept 2 channels (flow) instead of 3 (RGB)
            flow_feat.stem.conv = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            flow_feat_dim = 368
        elif 'rn50' in visual_arch:
            import torchvision.models as models
            flow_feat = models.resnet50(pretrained=pretrained)
            flow_feat_dim = flow_feat.fc.in_features
            flow_feat.fc = nn.Identity()
            # Modify first layer for flow (2 channels)
            original_conv = flow_feat.conv1
            flow_feat.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if pretrained:
                with torch.no_grad():
                    rgb_weight = original_conv.weight.data
                    flow_feat.conv1.weight.data = rgb_weight.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1)
            flow_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported visual_arch: {visual_arch}")
        
        # Add TSM modules if specified
        if '_tsm' in visual_arch:
            make_temporal_shift(rgb_feat, clip_len, is_gsm=False, step=1)
            make_temporal_shift(flow_feat, clip_len, is_gsm=False, step=1)
        
        self._rgb_feat = rgb_feat
        self._flow_feat = flow_feat
        self._rgb_feat_dim = rgb_feat_dim
        self._flow_feat_dim = flow_feat_dim
        
        # Temporal heads
        d_model = HIDDEN_DIM
        if temporal_arch == 'gru':
            self._rgb_head = nn.GRU(rgb_feat_dim, d_model, num_layers=1, batch_first=True)
            self._flow_head = nn.GRU(flow_feat_dim, d_model, num_layers=1, batch_first=True)
        elif temporal_arch == 'deeper_gru':
            self._rgb_head = nn.GRU(rgb_feat_dim, d_model, num_layers=3, batch_first=True)
            self._flow_head = nn.GRU(flow_feat_dim, d_model, num_layers=3, batch_first=True)
        else:
            raise NotImplementedError(temporal_arch)
        
        # Fusion layer
        if fusion_method == 'add':
            # Simple addition (requires same dimension)
            assert rgb_feat_dim == flow_feat_dim, "For 'add' fusion, RGB and Flow must have same feature dimension"
            self._fusion = None
            fused_dim = d_model
        elif fusion_method == 'concat':
            # Concatenation
            self._fusion = nn.Linear(d_model * 2, d_model)
            fused_dim = d_model
        elif fusion_method == 'weighted':
            # Weighted combination
            self._fusion = None
            self._rgb_weight = nn.Parameter(torch.ones(1) * 0.5)
            self._flow_weight = nn.Parameter(torch.ones(1) * 0.5)
            fused_dim = d_model
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")
        
        # Prediction heads
        self._coarse_pred = nn.Linear(fused_dim, 2)  # Event/Non-event
        self._fine_pred = nn.Linear(fused_dim, num_classes)  # Fine-grained classes
        
    def forward(self, frames, flow=None, skeleton=None):
        """
        Args:
            frames: [batch_size, clip_len, 3, H, W] - RGB frames
            flow: [batch_size, clip_len, 2, H, W] - optical flow
            skeleton: Not used
        """
        if frames is None or flow is None:
            raise ValueError("RGB+Flow Fusion model requires both RGB frames and flow input")
        
        batch_size, clip_len, rgb_channels, height, width = frames.shape
        _, _, flow_channels, _, _ = flow.shape
        
        # Extract RGB features (same as MD-FED)
        rgb_flat = frames.view(-1, rgb_channels, height, width)
        rgb_feat = self._rgb_feat(rgb_flat)  # [batch*clip_len, feat_dim] (RegNet with head.fc=Identity outputs 2D)
        rgb_feat = rgb_feat.reshape(batch_size, clip_len, -1)
        
        # Extract Flow features (same as MD-FED)
        flow_flat = flow.view(-1, flow_channels, height, width)
        flow_feat = self._flow_feat(flow_flat)  # [batch*clip_len, feat_dim] (RegNet with head.fc=Identity outputs 2D)
        flow_feat = flow_feat.reshape(batch_size, clip_len, -1)
        
        # Temporal modeling
        rgb_feat, _ = self._rgb_head(rgb_feat)  # [batch, clip_len, d_model]
        flow_feat, _ = self._flow_head(flow_feat)  # [batch, clip_len, d_model]
        
        # Fusion
        if self._fusion_method == 'add':
            fused_feat = rgb_feat + flow_feat
        elif self._fusion_method == 'concat':
            fused_feat = torch.cat([rgb_feat, flow_feat], dim=-1)  # [batch, clip_len, d_model*2]
            fused_feat = self._fusion(fused_feat)  # [batch, clip_len, d_model]
        elif self._fusion_method == 'weighted':
            # Weighted combination
            fused_feat = self._rgb_weight * rgb_feat + self._flow_weight * flow_feat
        
        # Predictions
        coarse_pred = self._coarse_pred(fused_feat)  # [batch, clip_len, 2]
        fine_pred = self._fine_pred(fused_feat)  # [batch, clip_len, num_classes]
        
        return coarse_pred, fine_pred
    
    def compute_loss(self, coarse_pred, fine_pred, coarse_label, fine_label):
        """
        计算损失函数
        """
        # Coarse loss (event detection)
        # Use weighted loss to handle class imbalance (event vs no-event)
        # Weight event class (class 1) 20x to encourage event prediction
        class_weights = torch.tensor([1.0, 20.0]).to(coarse_pred.device)
        coarse_loss = nn.CrossEntropyLoss(weight=class_weights)(
            coarse_pred.view(-1, 2),
            coarse_label.view(-1).long()
        )
        
        # Fine loss (fine-grained classification)
        # Only compute on event frames
        event_mask = (coarse_label > 0).float()
        fine_loss = nn.BCEWithLogitsLoss(reduction='none')(
            fine_pred,
            fine_label.float()
        )
        fine_loss = (fine_loss * event_mask.unsqueeze(-1)).sum() / (event_mask.sum() + 1e-6)
        
        total_loss = coarse_loss + fine_loss
        return total_loss, coarse_loss, fine_loss


def prepare_rgb_flow_data(manual_annotations_file, output_dir, dataset_name='ncaa-rally-rgb-flow', train_ratio=0.8):
    """
    准备 RGB + Flow 训练数据
    """
    print(f"Preparing RGB + Flow training data...")
    
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Total annotations: {len(annotations)}")
    
    # Shuffle and split
    random.seed(42)
    shuffled = annotations.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_annotations = shuffled[:split_idx]
    val_annotations = shuffled[split_idx:]
    
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    
    # Save files
    train_file = data_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    
    val_file = data_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    
    # Copy elements.txt
    elements_src = 'MD-FED/data/f3set-tennis-sub/elements.txt'
    if os.path.exists(elements_src):
        import shutil
        shutil.copy(elements_src, data_dir / 'elements.txt')
        print(f"Copied elements.txt")
    
    return str(data_dir)


def train_rgb_flow(args):
    """
    训练 RGB + Flow Fusion 模型
    """
    print(f"{'='*80}")
    print("Training RGB + Flow Fusion Model (Ablation Study)")
    print(f"{'='*80}\n")
    
    # Prepare data
    print("Step 1: Preparing data...")
    data_dir = prepare_rgb_flow_data(
        args.manual_annotations,
        args.data_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio
    )
    
    # Load classes
    elements_file = os.path.join(data_dir, 'elements.txt')
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Create model
    print("\nStep 2: Creating RGB + Flow Fusion model...")
    use_pretrained = not args.no_pretrained
    if use_pretrained:
        print("✅ 将使用 ImageNet 预训练权重（RGB 和 Flow）")
    else:
        print("⚠️  将从随机初始化开始训练（不使用预训练）")
    
    print(f"  Fusion method: {args.fusion_method}")
    
    model = RGB_Flow_Fusion_MD_FED(
        num_classes=len(classes),
        clip_len=args.clip_len,
        visual_arch=args.visual_arch,
        temporal_arch=args.temporal_arch,
        fusion_method=args.fusion_method,
        pretrained=use_pretrained
    ).cuda()
    
    print(f"Model configuration:")
    print(f"  Visual arch: {args.visual_arch}")
    print(f"  Temporal arch: {args.temporal_arch}")
    print(f"  Fusion method: {args.fusion_method}")
    print(f"  Clip len: {args.clip_len}")
    print(f"  Crop dim: {args.crop_dim}")
    
    # Create datasets
    print("\nStep 3: Creating datasets...")
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    epoch_num_frames = 10000
    dataset_len = epoch_num_frames // (args.clip_len * 2)
    
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, args.clip_len, dataset_len,
        is_eval=False, dilate_len=0, stage=3,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=None,
        crop_dim=args.crop_dim, stride=2
    )
    
    val_data = ActionSeqDataset(
        classes, val_json,
        args.frame_dir, args.clip_len, dataset_len // 4,
        is_eval=True,  # 验证时使用中心裁剪
        dilate_len=0, stage=3,
        num_samples=-1, flow_dir=args.flow_dir, pose_dir=None,
        crop_dim=args.crop_dim, stride=2
    )
    
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=args.batch_size,
        pin_memory=True, num_workers=4
    )
    
    # Use smaller batch size for validation to save memory
    val_batch_size = max(1, args.batch_size // 2) if args.batch_size > 1 else 1
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=val_batch_size,
        pin_memory=True, num_workers=4
    )
    
    # Load Stage 1 checkpoint if provided (optional, for RGB/Flow pretraining)
    if args.stage1_checkpoint:
        print(f"\n{'='*80}")
        print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        print(f"{'='*80}")
        
        if not os.path.exists(args.stage1_checkpoint):
            print(f"❌ Error: Stage 1 checkpoint not found: {args.stage1_checkpoint}")
            print(f"   Please check the path and try again.")
            sys.exit(1)
        
        checkpoint = torch.load(args.stage1_checkpoint, map_location='cuda')
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"✓ Checkpoint format: Full checkpoint (with metadata)")
            if 'epoch' in checkpoint:
                print(f"  Source epoch: {checkpoint['epoch'] + 1}")
            if 'val_loss' in checkpoint:
                print(f"  Source validation loss: {checkpoint['val_loss']:.4f}")
        else:
            # Assume it's a direct state dict
            model_state = checkpoint
            print(f"✓ Checkpoint format: Direct state dict")
        
        # Load weights (allow partial loading for compatibility)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in model_state.items() if k in model_dict and model_dict[k].shape == v.shape}
        skipped_params = len(model_state) - len(pretrained_dict)
        
        if skipped_params > 0:
            print(f"⚠️  Warning: Skipped {skipped_params} parameters (shape mismatch or not in current model)")
            print(f"   This is expected if the model architecture differs")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"\n✅ Successfully loaded Stage 1 checkpoint!")
        print(f"   Loaded: {len(pretrained_dict)}/{len(model_state)} parameters")
        print(f"{'='*80}\n")
    
    # Setup training
    print("\nStep 4: Setup training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    num_steps_per_epoch = len(train_loader)
    warm_up_epochs = 3
    cosine_epochs = args.num_epochs - warm_up_epochs
    
    lr_scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])
    
    # Training loop
    print("\nStep 5: Training...")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Clear GPU cache before training
    torch.cuda.empty_cache()
    
    # Print memory optimization info
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        print(f"⚠️  Using gradient accumulation: {args.gradient_accumulation_steps} steps")
        print(f"   Effective batch size: {effective_batch_size} (actual: {args.batch_size})")
    
    best_val_loss = float('inf')
    losses = []
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', leave=False)
        
        optimizer.zero_grad()
        accumulation_steps = 0
        
        for batch_idx, batch in enumerate(train_pbar):
            frames = batch['frame'].cuda()
            flow = batch['flow'].cuda()  # RGB + Flow uses both
            coarse_label = batch['coarse_label'].cuda()
            fine_label = batch['fine_label'].cuda()
            
            with torch.amp.autocast('cuda'):
                coarse_pred, fine_pred = model(frames=frames, flow=flow)
                loss, coarse_loss, fine_loss = model.compute_loss(
                    coarse_pred, fine_pred, coarse_label, fine_label
                )
                # Scale loss by accumulation steps
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            accumulation_steps += 1
            train_loss += loss.item() * args.gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if accumulation_steps >= args.gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                accumulation_steps = 0
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'acc_steps': f'{accumulation_steps}/{args.gradient_accumulation_steps}'
            })
        
        # Handle remaining gradients if any
        if accumulation_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                frames = batch['frame'].cuda()
                flow = batch['flow'].cuda()
                coarse_label = batch['coarse_label'].cuda()
                fine_label = batch['fine_label'].cuda()
                
                coarse_pred, fine_pred = model(frames=frames, flow=flow)
                loss, _, _ = model.compute_loss(
                    coarse_pred, fine_pred, coarse_label, fine_label
                )
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        # Clear GPU cache after validation
        torch.cuda.empty_cache()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'[Epoch {epoch+1}/{args.num_epochs}] Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {current_lr:.2e}')
        
        # Save checkpoint and check for improvement
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'  ✅ New best val loss: {val_loss:.5f} → Saving checkpoint...')
            # Save best model separately
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, 'best_model.pt')
            )
        else:
            patience_counter += 1
        
        losses.append({
            'epoch': epoch,
            'train': train_loss,
            'val': val_loss
        })
        
        store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
        
        # Save last checkpoint (for resuming training)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'losses': losses
            },
            os.path.join(args.save_dir, 'last_checkpoint.pt')
        )
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f'\n⚠️  Early stopping triggered after {patience_counter} epochs without improvement')
            break
    
    print(f'\n{"="*80}')
    print(f'Training Complete!')
    print(f'Best validation loss: {best_val_loss:.5f}')
    print(f'Checkpoints saved to: {args.save_dir}')
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train RGB + Flow Fusion model for ablation study'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        default='manual_annotations.json',
        help='Path to manual annotations'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Path to extracted frames'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        required=True,
        help='Path to extracted optical flows'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='rgb_flow_data',
        help='Directory to save prepared data'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally-rgb-flow',
        help='Dataset name'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default='rny002_tsm',
        choices=['rny002_tsm', 'rny002', 'rn50_tsm', 'rn50'],
        help='Visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru'],
        help='Temporal modeling architecture (default: gru)'
    )
    parser.add_argument(
        '--fusion_method',
        type=str,
        default='add',
        choices=['add', 'concat', 'weighted'],
        help='Fusion method: add (simple addition), concat (concatenation), weighted (learnable weights)'
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='不使用预训练权重'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='Clip length'
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=224,
        help='Crop dimension for images (default: 224)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=10,
        help='Early stopping patience (number of epochs without improvement, default: 10)'
    )
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=0.001,
        help='Minimum change to qualify as improvement (default: 0.001)'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (default: 1). Use this to simulate larger batch sizes when GPU memory is limited.'
    )
    parser.add_argument(
        '--stage1_checkpoint',
        type=str,
        default=None,
        help='Path to Stage 1 checkpoint for initialization (optional)'
    )
    
    args = parser.parse_args()
    
    train_rgb_flow(args)


if __name__ == '__main__':
    main()
