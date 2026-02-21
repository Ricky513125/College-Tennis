#!/usr/bin/env python3
"""
Train RGB + Flow + Skeleton Fusion model for ablation study.

This script trains a model that directly fuses RGB, Flow, and Skeleton features
together, WITHOUT using distillation. This is an ablation study to compare with
MD-FED (which uses distillation in Stage 2).

According to the paper: "fusing RGB, optical flow, and skeleton leads to 
significantly lower performance, with edit scores dropping by up to 21.8%, 
highlighting the superiority of distillation over direct fusion."

Usage:
    python train_rgb_flow_skeleton_fusion.py \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --flow_dir /path/to/flows \
        --pose_dir /path/to/skeletons \
        --save_dir ./rgb_flow_skeleton_fusion_outputs \
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
from model.stgcn import STGCN
from util.dataset import load_classes
from util.io import load_json, store_json
from dataset.input_process import ActionSeqDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import random
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

HIDDEN_DIM = 368


def collate_fn_skeleton_padding(batch):
    """
    Custom collate function to handle variable number of people in skeleton data.
    Pads or truncates to 2 people (TENNIS_SINGLES_NUM_PEOPLE).
    """
    TENNIS_SINGLES_NUM_PEOPLE = 2
    normalized_batch = []
    for item in batch:
        if 'skeleton' in item and item['skeleton'] is not None:
            skeleton = item['skeleton']
            if len(skeleton.shape) == 4:
                T, num_people, num_joints, coords = skeleton.shape
                if num_people > TENNIS_SINGLES_NUM_PEOPLE:
                    skeleton = skeleton[:, :TENNIS_SINGLES_NUM_PEOPLE, :, :]
                elif num_people < TENNIS_SINGLES_NUM_PEOPLE:
                    padding = torch.zeros(
                        T, TENNIS_SINGLES_NUM_PEOPLE - num_people, num_joints, coords,
                        dtype=skeleton.dtype, device=skeleton.device
                    )
                    skeleton = torch.cat([skeleton, padding], dim=1)
                item['skeleton'] = skeleton
        normalized_batch.append(item)
    from torch.utils.data._utils.collate import default_collate
    return default_collate(normalized_batch)


class RGB_Flow_Skeleton_Fusion_MD_FED(nn.Module):
    """
    直接融合 RGB、Flow 和 Skeleton 特征的模型（不使用蒸馏）
    用于消融实验：对比直接融合 vs 蒸馏
    """
    def __init__(self, num_classes, clip_len, visual_arch='rny002_tsm', 
                 skeleton_arch='stgcn++', temporal_arch='gru', 
                 fusion_method='concat', pretrained=True):
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
        
        # Skeleton feature extractor (STGCN++)
        if 'stgcn++' in skeleton_arch:
            sk_feat = STGCN(
                in_channels=2, 
                data_bn_type='MVC', 
                gcn_adaptive='init', 
                gcn_with_res=True,
                tcn_type='mstcn', 
                graph_cfg=dict(layout='coco', mode='spatial')
            )
            sk_feat_dim = 256
        elif 'stgcn' in skeleton_arch:
            sk_feat = STGCN(
                in_channels=2, 
                data_bn_type='MVC', 
                graph_cfg=dict(layout='coco', mode='stgcn_spatial')
            )
            sk_feat_dim = 256
        else:
            raise ValueError(f"Unsupported skeleton_arch: {skeleton_arch}")
        
        # Add TSM modules if specified
        if '_tsm' in visual_arch:
            make_temporal_shift(rgb_feat, clip_len, is_gsm=False, step=1)
            make_temporal_shift(flow_feat, clip_len, is_gsm=False, step=1)
        
        self._rgb_feat = rgb_feat
        self._flow_feat = flow_feat
        self._sk_feat = sk_feat
        self._rgb_feat_dim = rgb_feat_dim
        self._flow_feat_dim = flow_feat_dim
        self._sk_feat_dim = sk_feat_dim
        
        # Temporal heads
        d_model = HIDDEN_DIM
        if temporal_arch == 'gru':
            self._rgb_head = nn.GRU(rgb_feat_dim, d_model, num_layers=1, batch_first=True)
            self._flow_head = nn.GRU(flow_feat_dim, d_model, num_layers=1, batch_first=True)
            self._sk_head = nn.GRU(sk_feat_dim, d_model, num_layers=1, batch_first=True)
        elif temporal_arch == 'deeper_gru':
            self._rgb_head = nn.GRU(rgb_feat_dim, d_model, num_layers=3, batch_first=True)
            self._flow_head = nn.GRU(flow_feat_dim, d_model, num_layers=3, batch_first=True)
            self._sk_head = nn.GRU(sk_feat_dim, d_model, num_layers=3, batch_first=True)
        else:
            raise NotImplementedError(temporal_arch)
        
        # Fusion layer
        if fusion_method == 'add':
            # Simple addition (requires same dimension)
            assert rgb_feat_dim == flow_feat_dim == sk_feat_dim, \
                "For 'add' fusion, all modalities must have same feature dimension"
            self._fusion = None
            fused_dim = d_model
        elif fusion_method == 'concat':
            # Concatenation
            self._fusion = nn.Linear(d_model * 3, d_model)  # RGB + Flow + Skeleton
            fused_dim = d_model
        elif fusion_method == 'weighted':
            # Weighted combination
            self._fusion = None
            self._rgb_weight = nn.Parameter(torch.ones(1) * 0.33)
            self._flow_weight = nn.Parameter(torch.ones(1) * 0.33)
            self._sk_weight = nn.Parameter(torch.ones(1) * 0.34)
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
            skeleton: [batch_size, clip_len, num_person, num_joints, 2] - skeleton data
        """
        if frames is None or flow is None or skeleton is None:
            raise ValueError("RGB+Flow+Skeleton Fusion model requires all three modalities")
        
        batch_size, clip_len, rgb_channels, height, width = frames.shape
        _, _, flow_channels, _, _ = flow.shape
        
        # Extract RGB features
        rgb_flat = frames.view(-1, rgb_channels, height, width)
        rgb_feat = self._rgb_feat(rgb_flat)  # [batch*clip_len, feat_dim]
        rgb_feat = rgb_feat.reshape(batch_size, clip_len, -1)
        
        # Extract Flow features
        flow_flat = flow.view(-1, flow_channels, height, width)
        flow_feat = self._flow_feat(flow_flat)  # [batch*clip_len, feat_dim]
        flow_feat = flow_feat.reshape(batch_size, clip_len, -1)
        
        # Extract Skeleton features
        # Handle skeleton shape
        if len(skeleton.shape) == 4:
            # [batch_size, clip_len, num_joints, 2] -> [batch_size, clip_len, 1, num_joints, 2]
            skeleton = skeleton.unsqueeze(2)
        
        batch_size, clip_len, num_person, num_joints, num_coords = skeleton.shape
        
        # STGCN expects: [N, M, T, V, C]
        skeleton_transposed = skeleton.transpose(1, 2)  # [batch_size, M, clip_len, V, C]
        sk_feat = self._sk_feat(skeleton_transposed)  # [batch_size, M, feat_dim, T', V'] or similar
        
        # Aggregate over person dimension and spatial dimensions (same as train_stgcn_comparison.py)
        # sk_feat shape after STGCN: [batch_size, M, feat_dim, T', V'] or similar
        sk_feat = sk_feat.mean(dim=1)  # [batch_size, feat_dim, T', V'] - average over persons
        sk_feat = sk_feat.mean(dim=-1)  # [batch_size, feat_dim, T'] - average over joints
        sk_feat = sk_feat.transpose(1, 2)  # [batch_size, T', feat_dim]
        
        # Reshape to [batch_size, clip_len, feat_dim]
        # If T' != clip_len, interpolate
        if sk_feat.shape[1] != clip_len:
            sk_feat = torch.nn.functional.interpolate(
                sk_feat.transpose(1, 2),  # [batch_size, feat_dim, T']
                size=clip_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch_size, clip_len, feat_dim]
        
        # Ensure correct feature dimension
        if sk_feat.shape[2] != self._sk_feat_dim:
            # If feature dimension doesn't match, use a projection layer
            if not hasattr(self, '_sk_feat_proj'):
                self._sk_feat_proj = nn.Linear(sk_feat.shape[2], self._sk_feat_dim).to(sk_feat.device)
            sk_feat = self._sk_feat_proj(sk_feat)
        
        # Temporal modeling
        rgb_feat, _ = self._rgb_head(rgb_feat)  # [batch, clip_len, d_model]
        flow_feat, _ = self._flow_head(flow_feat)  # [batch, clip_len, d_model]
        sk_feat, _ = self._sk_head(sk_feat)  # [batch, clip_len, d_model]
        
        # Fusion
        if self._fusion_method == 'add':
            fused_feat = rgb_feat + flow_feat + sk_feat
        elif self._fusion_method == 'concat':
            fused_feat = torch.cat([rgb_feat, flow_feat, sk_feat], dim=-1)  # [batch, clip_len, d_model*3]
            fused_feat = self._fusion(fused_feat)  # [batch, clip_len, d_model]
        elif self._fusion_method == 'weighted':
            fused_feat = self._rgb_weight * rgb_feat + self._flow_weight * flow_feat + self._sk_weight * sk_feat
        
        # Predictions
        coarse_pred = self._coarse_pred(fused_feat)  # [batch, clip_len, 2]
        fine_pred = self._fine_pred(fused_feat)  # [batch, clip_len, num_classes]
        
        return coarse_pred, fine_pred
    
    def compute_loss(self, coarse_pred, fine_pred, coarse_label, fine_label):
        """
        Compute loss for direct fusion (no distillation).
        Uses standard classification losses.
        """
        # Coarse-grained loss (weighted CrossEntropyLoss for class imbalance)
        class_weights = torch.tensor([1.0, 20.0]).to(coarse_pred.device)  # Weight event class (1) 20x
        coarse_loss = nn.CrossEntropyLoss(weight=class_weights)(
            coarse_pred.view(-1, 2),
            coarse_label.view(-1).long()
        )
        
        # Fine-grained loss (BCEWithLogitsLoss for multi-label)
        # Only compute on event frames
        event_mask = (coarse_label > 0).float()
        fine_loss = nn.BCEWithLogitsLoss(reduction='none')(
            fine_pred,
            fine_label.float()
        )
        fine_loss = (fine_loss * event_mask.unsqueeze(-1)).sum() / (event_mask.sum() + 1e-6)
        
        total_loss = coarse_loss + fine_loss
        return total_loss, coarse_loss, fine_loss


def prepare_stage2_data(manual_annotations_file, output_dir, dataset_name='ncaa-rally', train_ratio=0.8):
    """
    准备 Stage 2 训练数据（RGB + Flow + Skeleton）
    """
    print(f"Preparing Stage 2 data for RGB+Flow+Skeleton Fusion ablation...")
    
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
    
    print(f"  Train: {len(train_annotations)} rallies")
    print(f"  Val: {len(val_annotations)} rallies")
    
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
        print(f"Copied elements.txt to {data_dir / 'elements.txt'}")
    
    return str(data_dir)


def train_epoch(model, loader, optimizer, scaler, device, gradient_accumulation_steps=1):
    """Training epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_coarse_loss = 0.0
    epoch_fine_loss = 0.0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        frames = loader.dataset.load_frame_gpu(batch, device)
        flow = loader.dataset.load_flow_gpu(batch, device)
        skeleton = loader.dataset.load_skeleton_gpu(batch, device)
        
        coarse_label = batch['coarse_label'].to(device)
        fine_label = batch['fine_label'].to(device)
        
        with torch.amp.autocast('cuda'):
            coarse_pred, fine_pred = model(frames, flow, skeleton)
            total_loss, coarse_loss, fine_loss = model.compute_loss(
                coarse_pred, fine_pred, coarse_label, fine_label
            )
            total_loss = total_loss / gradient_accumulation_steps
        
        scaler.scale(total_loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += total_loss.item() * gradient_accumulation_steps
        epoch_coarse_loss += coarse_loss.item()
        epoch_fine_loss += fine_loss.item()
    
    return epoch_loss / len(loader), epoch_coarse_loss / len(loader), epoch_fine_loss / len(loader)


def validate_epoch(model, loader, device):
    """Validation epoch"""
    model.eval()
    epoch_loss = 0.0
    epoch_coarse_loss = 0.0
    epoch_fine_loss = 0.0
    
    correct_coarse = 0
    total_coarse = 0
    correct_fine = 0
    total_fine = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            frames = loader.dataset.load_frame_gpu(batch, device)
            flow = loader.dataset.load_flow_gpu(batch, device)
            skeleton = loader.dataset.load_skeleton_gpu(batch, device)
            
            coarse_label = batch['coarse_label'].to(device)
            fine_label = batch['fine_label'].to(device)
            
            with torch.amp.autocast('cuda'):
                coarse_pred, fine_pred = model(frames, flow, skeleton)
                total_loss, coarse_loss, fine_loss = model.compute_loss(
                    coarse_pred, fine_pred, coarse_label, fine_label
                )
            
            epoch_loss += total_loss.item()
            epoch_coarse_loss += coarse_loss.item()
            epoch_fine_loss += fine_loss.item()
            
            # Accuracy
            coarse_pred_class = coarse_pred.argmax(dim=-1)
            correct_coarse += (coarse_pred_class == coarse_label).sum().item()
            total_coarse += coarse_label.numel()
            
            fine_pred_sigmoid = torch.sigmoid(fine_pred) > 0.5
            correct_fine += (fine_pred_sigmoid == fine_label.bool()).sum().item()
            total_fine += fine_label.numel()
    
    coarse_acc = 100.0 * correct_coarse / total_coarse if total_coarse > 0 else 0.0
    fine_acc = 100.0 * correct_fine / total_fine if total_fine > 0 else 0.0
    
    return (
        epoch_loss / len(loader),
        epoch_coarse_loss / len(loader),
        epoch_fine_loss / len(loader),
        coarse_acc,
        fine_acc
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train RGB + Flow + Skeleton Fusion model (Direct Fusion, No Distillation)'
    )
    parser.add_argument('--manual_annotations', type=str, required=True,
                       help='Path to manual_annotations.json')
    parser.add_argument('--frame_dir', type=str, required=True,
                       help='Directory containing RGB frames')
    parser.add_argument('--flow_dir', type=str, required=True,
                       help='Directory containing optical flow files')
    parser.add_argument('--pose_dir', type=str, required=True,
                       help='Directory containing skeleton (pose) files')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save checkpoints and logs')
    parser.add_argument('--data_dir', type=str, default='./md_fed_data',
                       help='Directory to save prepared data')
    parser.add_argument('--dataset_name', type=str, default='ncaa-rally',
                       help='Dataset name for prepared data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    
    # Model arguments
    parser.add_argument('--visual_arch', type=str, default='rny002_tsm',
                       choices=['rny002_tsm', 'rn50_tsm'],
                       help='Visual feature extractor architecture')
    parser.add_argument('--skeleton_arch', type=str, default='stgcn++',
                       choices=['stgcn++', 'stgcn'],
                       help='Skeleton feature extractor architecture')
    parser.add_argument('--temporal_arch', type=str, default='gru',
                       choices=['gru', 'deeper_gru'],
                       help='Temporal modeling architecture')
    parser.add_argument('--fusion_method', type=str, default='concat',
                       choices=['add', 'concat', 'weighted'],
                       help='Fusion method for RGB+Flow+Skeleton features')
    
    # Training arguments
    parser.add_argument('--crop_dim', type=int, default=224,
                       help='Crop dimension for frames')
    parser.add_argument('--clip_len', type=int, default=96,
                       help='Number of frames per clip')
    parser.add_argument('--stride', type=int, default=2,
                       help='Stride for frame sampling')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--warm_up_epochs', type=int, default=10,
                       help='Number of warm-up epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained weights')
    
    # Checkpoint loading
    parser.add_argument('--stage1_model_dir', type=str, default=None,
                       help='Directory containing Stage 1 skeleton checkpoint (best_model.pt)')
    parser.add_argument('--stage2_checkpoint', type=str, default=None,
                       help='Path to MD-FED Stage 2 checkpoint (optional, for fair comparison)')
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                       help='Minimum delta for early stopping')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("⚠️  消融实验：RGB + Flow + Skeleton 直接融合（不使用蒸馏）")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    data_dir = prepare_stage2_data(
        args.manual_annotations,
        args.data_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio
    )
    
    # Step 2: Load classes
    print("\nStep 2: Loading classes...")
    elements_file = os.path.join(data_dir, 'elements.txt')
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Step 3: Create datasets
    print("\nStep 3: Creating datasets...")
    train_json = os.path.join(data_dir, 'train.json')
    val_json = os.path.join(data_dir, 'val.json')
    
    epoch_num_frames = 100000  # Large dataset for Stage 2
    dataset_len = epoch_num_frames // (args.clip_len * args.stride)
    
    dataset_kwargs = {
        'crop_dim': args.crop_dim,
        'stride': args.stride
    }
    
    train_dataset = ActionSeqDataset(
        classes,
        train_json,
        args.frame_dir,
        args.clip_len,
        dataset_len,
        is_eval=False,
        dilate_len=0,
        stage=2,
        num_samples=-1,
        flow_dir=args.flow_dir,
        pose_dir=args.pose_dir,
        **dataset_kwargs
    )
    
    val_dataset = ActionSeqDataset(
        classes,
        val_json,
        args.frame_dir,
        args.clip_len,
        dataset_len // 4,
        is_eval=True,
        dilate_len=0,
        stage=2,
        num_samples=-1,
        flow_dir=args.flow_dir,
        pose_dir=args.pose_dir,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_skeleton_padding
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn_skeleton_padding
    )
    
    # Step 4: Create model
    print("\nStep 4: Creating RGB+Flow+Skeleton Fusion model...")
    use_pretrained = not args.no_pretrained
    if use_pretrained:
        print("✅ 将使用 ImageNet 预训练权重（RGB 和 Flow）")
    else:
        print("⚠️  将从随机初始化开始训练（不使用预训练）")
    
    print(f"  Fusion method: {args.fusion_method}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGB_Flow_Skeleton_Fusion_MD_FED(
        num_classes=len(classes),
        clip_len=args.clip_len,
        visual_arch=args.visual_arch,
        skeleton_arch=args.skeleton_arch,
        temporal_arch=args.temporal_arch,
        fusion_method=args.fusion_method,
        pretrained=use_pretrained
    ).to(device)
    
    print(f"Model configuration:")
    print(f"  Visual arch: {args.visual_arch}")
    print(f"  Skeleton arch: {args.skeleton_arch}")
    print(f"  Temporal arch: {args.temporal_arch}")
    print(f"  Fusion method: {args.fusion_method}")
    print(f"  Clip len: {args.clip_len}")
    print(f"  Crop dim: {args.crop_dim}")
    
    # Step 5: Load Stage 1 checkpoint for skeleton initialization
    if args.stage1_model_dir:
        stage1_checkpoint = os.path.join(args.stage1_model_dir, 'best_model.pt')
        if os.path.exists(stage1_checkpoint):
            print(f"\nLoading Stage 1 checkpoint: {stage1_checkpoint}")
            checkpoint = torch.load(stage1_checkpoint, map_location=device)
            
            # Extract skeleton weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            for key, value in state_dict.items():
                # Map Stage 1 skeleton weights to our model
                if key.startswith('_sk_feat.') or key.startswith('_sk_head.'):
                    if key in model_dict and model_dict[key].shape == value.shape:
                        pretrained_dict[key] = value
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"✓ Loaded {len(pretrained_dict)} skeleton parameters from Stage 1")
        else:
            print(f"⚠️  Stage 1 checkpoint not found: {stage1_checkpoint}")
    
    # Step 6: Load Stage 2 checkpoint (optional, for fair comparison)
    if args.stage2_checkpoint:
        print(f"\nLoading MD-FED Stage 2 checkpoint: {args.stage2_checkpoint}")
        print("⚠️  这将加载 RGB 和 Flow 特征提取器权重（用于公平对比）")
        
        if not os.path.exists(args.stage2_checkpoint):
            print(f"❌ Error: Stage 2 checkpoint not found: {args.stage2_checkpoint}")
            sys.exit(1)
        
        checkpoint = torch.load(args.stage2_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            stage2_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            stage2_state = checkpoint['state_dict']
        else:
            stage2_state = checkpoint
        
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        for stage2_key, stage2_value in stage2_state.items():
            # Remove _model. prefix if present
            if stage2_key.startswith('_model.'):
                our_key = stage2_key[7:]
            else:
                our_key = stage2_key
            
            # Only load RGB and Flow feature extractors and temporal heads
            if our_key in model_dict:
                if model_dict[our_key].shape == stage2_value.shape:
                    pretrained_dict[our_key] = stage2_value
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✓ Loaded {len(pretrained_dict)} RGB/Flow parameters from Stage 2")
    
    # Step 7: Setup training
    print("\nStep 7: Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warm_up_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warm_up_epochs, eta_min=1e-6
    )
    lr_scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
    
    # Step 8: Training loop
    print("\nStep 8: Starting training...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'visual_arch': args.visual_arch,
        'skeleton_arch': args.skeleton_arch,
        'temporal_arch': args.temporal_arch,
        'fusion_method': args.fusion_method,
        'num_classes': len(classes),
        'clip_len': args.clip_len,
        'crop_dim': args.crop_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'stage': 2,
        'stride': args.stride,
    }
    store_json(save_dir / 'config.json', config, pretty=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    losses = {'train': [], 'val': []}
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Training
        train_loss, train_coarse_loss, train_fine_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, args.gradient_accumulation_steps
        )
        
        # Validation
        val_loss, val_coarse_loss, val_fine_loss, val_coarse_acc, val_fine_acc = validate_epoch(
            model, val_loader, device
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Logging
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} (Coarse: {train_coarse_loss:.4f}, Fine: {train_fine_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Coarse: {val_coarse_loss:.4f}, Fine: {val_fine_loss:.4f})")
        print(f"  Val Acc: Coarse={val_coarse_acc:.2f}%, Fine={val_fine_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Save last checkpoint
        last_checkpoint_path = save_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {args.early_stop_patience} epochs without improvement")
            print(f"   Best val loss: {best_val_loss:.4f}")
            break
        
        # Save loss history
        store_json(save_dir / 'loss.json', losses, pretty=True)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
