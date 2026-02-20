#!/usr/bin/env python3
"""
Train Stage 2 with RGB alone as Student (Ablation Study).

This is an ablation experiment where only RGB learns from Skeleton features,
and Flow is frozen (not updated).

Original MD-FED Stage 2:
- Teacher: Skeleton
- Students: RGB, Flow
- Loss: MSE(RGB_feat, Skeleton_feat) + MSE(Flow_feat, Skeleton_feat)

This ablation (RGB alone as Student):
- Teacher: Skeleton (pre-trained in Stage 1)
- Student: RGB only
- Flow: Frozen (not updated)
- Loss: MSE(RGB_feat, Skeleton_feat)
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


def collate_fn_skeleton_padding(batch):
    """
    Custom collate function to handle skeleton data for tennis singles.
    For tennis singles, we only need 2 people (the two players).
    - If more than 2 people detected, keep only the first 2
    - If less than 2 people, pad with zeros to 2
    
    Skeleton shape: [T, num_people, num_joints, 2]
    """
    TENNIS_SINGLES_NUM_PEOPLE = 2  # Tennis singles has 2 players
    
    # Normalize all skeleton tensors to exactly 2 people
    normalized_batch = []
    for item in batch:
        if 'skeleton' in item and item['skeleton'] is not None:
            skeleton = item['skeleton']
            if len(skeleton.shape) == 4:  # [T, num_people, num_joints, 2]
                T, num_people, num_joints, coords = skeleton.shape
                
                if num_people > TENNIS_SINGLES_NUM_PEOPLE:
                    # Keep only first 2 people (tennis singles)
                    skeleton = skeleton[:, :TENNIS_SINGLES_NUM_PEOPLE, :, :]
                elif num_people < TENNIS_SINGLES_NUM_PEOPLE:
                    # Pad with zeros to 2 people
                    padding = torch.zeros(T, TENNIS_SINGLES_NUM_PEOPLE - num_people, num_joints, coords, 
                                        dtype=skeleton.dtype, device=skeleton.device)
                    skeleton = torch.cat([skeleton, padding], dim=1)
                
                item['skeleton'] = skeleton
        normalized_batch.append(item)
    
    # Use default collate for the rest
    from torch.utils.data._utils.collate import default_collate
    return default_collate(normalized_batch)


class MD_FED_RGB_Student(MD_FED):
    """
    Modified MD-FED with RGB alone as student network.
    In Stage 2, only RGB learns from Skeleton features (pre-trained in Stage 1).
    Flow parameters are frozen and not updated.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Freeze Flow parameters
        self._freeze_flow_parameters()
    
    def _freeze_flow_parameters(self):
        """Freeze Flow parameters so they don't get updated."""
        # Handle both regular model and DataParallel wrapped model
        model = self._model
        if hasattr(model, 'module'):  # DataParallel wrapped
            model = model.module
        
        if hasattr(model, '_flow_feat'):
            for param in model._flow_feat.parameters():
                param.requires_grad = False
            print("✓ Frozen Flow feature extractor parameters")
        if hasattr(model, '_flow_head'):
            for param in model._flow_head.parameters():
                param.requires_grad = False
            print("✓ Frozen Flow head parameters")
    
    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5):
        """
        Modified epoch function for RGB alone as student distillation.
        Only RGB learns from Skeleton features.
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

                    # stage 2: multimodal distillation with RGB alone as student
                    if self._stage == 2:
                        # Get features from all modalities
                        _, _, rgb_feat, flow_feat, sk_feat = self._model(frame, flow, skeleton)
                        
                        # RGB alone as student: Only RGB learns from Skeleton (teacher)
                        # Flow is frozen, so we don't compute flow2sk_loss
                        rgb2sk_loss = F.mse_loss(rgb_feat, sk_feat)
                        
                        loss += rgb2sk_loss
                        
                        # Log losses for monitoring
                        if batch_idx == 0 and mode == "Training":
                            print(f"\n[RGB alone as Student] RGB→Skeleton loss: {rgb2sk_loss.item():.6f}")
                            print(f"  (Flow parameters are frozen, not updated)")

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
    print(f"Preparing Stage 2 data for RGB alone as Student ablation...")
    
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
    elements_dst = data_dir / 'elements.txt'
    if elements_src.exists():
        import shutil
        shutil.copy(elements_src, elements_dst)
        print(f"Copied elements.txt to {elements_dst}")
    else:
        print(f"Warning: {elements_src} not found, please create elements.txt manually")
    
    return str(data_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Train Stage 2 with RGB alone as Student (Ablation Study)'
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
        help='Directory to save Stage 2 checkpoints (e.g., ./md_fed_outputs/stage2_rgb_student)'
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
    print("Stage 2 Training: RGB alone as Student (Ablation Study)")
    print("=" * 80)
    print("⚠️  消融实验：只有 RGB 学习 Skeleton 特征，Flow 参数被冻结")
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
    
    train_dataset = ActionSeqDataset(
        train_json,
        args.frame_dir,
        args.flow_dir,
        args.pose_dir,
        classes,
        clip_len=args.clip_len,
        dataset_len=dataset_len,
        randomize=True,
        **dataset_kwargs
    )
    
    val_dataset = ActionSeqDataset(
        val_json,
        args.frame_dir,
        args.flow_dir,
        args.pose_dir,
        classes,
        clip_len=args.clip_len,
        dataset_len=dataset_len // 4,
        randomize=False,
        **dataset_kwargs
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Step 4: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_skeleton_padding
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_skeleton_padding
    )
    
    # Step 5: Create model
    print("\nStep 3: Creating model...")
    model = MD_FED_RGB_Student(
        num_classes=len(classes),
        visual_arch=args.visual_arch,
        skeleton_arch=args.skeleton_arch,
        temporal_arch=args.temporal_arch,
        clip_len=args.clip_len,
        stage=2,
        device='cuda'
    )
    
    # Step 6: Load Stage 1 checkpoint for skeleton initialization
    stage1_checkpoint = os.path.join(args.stage1_model_dir, 'best_model.pt')
    if os.path.exists(stage1_checkpoint):
        print(f"\nLoading Stage 1 checkpoint: {stage1_checkpoint}")
        checkpoint = torch.load(stage1_checkpoint, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        # Load skeleton parameters only
        model_dict = model._model.state_dict()
        skeleton_keys = [k for k in model_state.keys() if 'skeleton' in k.lower() or '_sk_' in k]
        loaded_keys = []
        for key in skeleton_keys:
            if key in model_dict:
                model_dict[key] = model_state[key]
                loaded_keys.append(key)
        
        model._model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(loaded_keys)} skeleton parameters from Stage 1")
    else:
        print(f"Warning: Stage 1 checkpoint not found at {stage1_checkpoint}")
        print("Skeleton will be initialized randomly")
    
    # Step 7: Create optimizer (only for RGB parameters, Flow is frozen)
    print("\nStep 4: Creating optimizer...")
    # Get only trainable parameters (RGB and Skeleton, Flow is frozen)
    trainable_params = [p for p in model._model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in model._model.parameters() if not p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Step 8: Create learning rate scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * args.warm_up_epochs
    
    lr_scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps, eta_min=1e-6)
    ])
    
    # Step 9: Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Step 10: Training loop
    print("\nStep 5: Starting training...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = model.epoch(
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            acc_grad_iter=args.acc_grad_iter
        )
        
        # Validate
        val_loss = model.epoch(val_loader, optimizer=None)
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model._model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Save last checkpoint
        last_checkpoint_path = save_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Step 11: Save config for Stage 3
    config = {
        'num_classes': len(classes),
        'visual_arch': args.visual_arch,
        'skeleton_arch': args.skeleton_arch,
        'temporal_arch': args.temporal_arch,
        'clip_len': args.clip_len,
        'stage': 2,
        'ablation': 'rgb_alone_as_student',
        'description': 'RGB alone learns from Skeleton, Flow is frozen'
    }
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Saved config to {config_path}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
