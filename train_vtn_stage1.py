"""
VTN Stage 1: Pre-training on F3Set Dataset

This script pre-trains the VTN model on the F3Set dataset, similar to MD-FED Stage 1.
After Stage 1 training, the checkpoint can be used as initialization for Stage 3 fine-tuning
on manual_annotations.json.

Usage:
    python train_vtn_stage1.py \
        --frame_dir /path/to/f3set_frames \
        --save_dir ./vtn_outputs/stage1 \
        --crop_dim 224 \
        --clip_len 96 \
        --batch_size 4 \
        --num_epochs 50 \
        --vtn_spatial_size small \
        --vtn_temporal_type longformer
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add MD-FED to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MD-FED'))

from model.vtn import VTN
from dataset.input_process import ActionSeqDataset
from util.dataset import load_classes
from util.io import store_json


class VTNModel(nn.Module):
    """VTN wrapper for Stage 1 training"""
    
    def __init__(self, num_classes, clip_len=96, crop_dim=224, 
                 vtn_spatial_size='small', vtn_temporal_type='longformer',
                 pretrained=True, pretrained_path=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.clip_len = clip_len
        
        # VTN backbone
        self.vtn = VTN(
            frames=clip_len,
            num_classes=1000,  # Not used, we'll use our own heads
            img_size=crop_dim,
            patch_size=16,
            spatial_size=vtn_spatial_size,
            temporal_type=vtn_temporal_type,
            pretrained=pretrained,
            pretrained_path=pretrained_path
        )
        
        # Get VTN output dimension based on spatial_size
        spatial_size_to_dim = {
            'tiny': 192,
            'small': 384,
            'base': 768,
            'large': 1024
        }
        vtn_dim = spatial_size_to_dim.get(vtn_spatial_size, 384)
        
        # Prediction heads
        self.temporal_head = nn.GRU(
            input_size=vtn_dim,
            hidden_size=276,
            num_layers=1,
            batch_first=True
        )
        
        self.coarse_classifier = nn.Linear(276, 2)  # Event vs non-event
        self.fine_classifier = nn.Linear(276, num_classes)  # Fine-grained classes
        
    def forward(self, frames):
        """
        Args:
            frames: [batch_size, clip_len, 3, H, W]
        Returns:
            coarse_pred: [batch_size, clip_len, 2]
            fine_pred: [batch_size, clip_len, num_classes]
        """
        batch_size = frames.size(0)
        clip_len = frames.size(1)
        
        # Reshape to [batch_size*clip_len, 3, H, W] for VTN
        frames_flat = frames.view(batch_size * clip_len, *frames.shape[2:])
        
        # VTN feature extraction (returns [batch_size*clip_len, vtn_dim])
        features = self.vtn(frames_flat)
        
        # Reshape back to [batch_size, clip_len, vtn_dim]
        features = features.view(batch_size, clip_len, -1)
        
        # Temporal modeling
        temporal_features, _ = self.temporal_head(features)  # [batch_size, clip_len, 276]
        
        # Classification
        coarse_pred = self.coarse_classifier(temporal_features)
        fine_pred = self.fine_classifier(temporal_features)
        
        return coarse_pred, fine_pred


def train_epoch(model, train_loader, optimizer, scaler, device, fg_weight=5):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_coarse_correct = 0
    total_coarse_samples = 0
    total_fine_correct = 0
    total_fine_samples = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frame'].to(device)
        coarse_label = batch['coarse_label'].to(device)
        fine_label = batch['fine_label'].to(device)
        coarse_mask = batch['coarse_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            coarse_pred, fine_pred = model(frames)
            
            # Coarse-grained loss (event detection)
            coarse_loss = nn.functional.cross_entropy(
                coarse_pred.reshape(-1, 2),
                coarse_label.reshape(-1),
                reduction='none'
            )
            coarse_loss = (coarse_loss * coarse_mask.reshape(-1)).mean()
            
            # Fine-grained loss (element classification) - multi-label
            # fine_label is [batch_size, clip_len, num_classes] with binary values
            fine_loss = nn.functional.binary_cross_entropy_with_logits(
                fine_pred,
                fine_label.float(),
                reduction='none'
            )
            # Only compute fine loss on event frames
            event_mask = (coarse_label == 1).float().unsqueeze(-1)  # [batch, clip_len, 1]
            fine_loss = (fine_loss * event_mask * coarse_mask.unsqueeze(-1)).sum()
            fine_loss = fine_loss / (event_mask * coarse_mask.unsqueeze(-1)).sum().clamp(min=1)
            
            # Total loss
            loss = coarse_loss + fg_weight * fine_loss
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        total_loss += loss.item()
        
        with torch.no_grad():
            # Coarse accuracy
            coarse_pred_labels = coarse_pred.argmax(dim=-1)
            coarse_correct = ((coarse_pred_labels == coarse_label) * coarse_mask).sum()
            total_coarse_correct += coarse_correct.item()
            total_coarse_samples += coarse_mask.sum().item()
            
            # Fine accuracy (only on event frames) - multi-label with threshold 0.5
            fine_pred_binary = (torch.sigmoid(fine_pred) > 0.5).float()
            event_mask = (coarse_label == 1).float().unsqueeze(-1)
            # Calculate per-frame accuracy (all elements correct in a frame = 1.0)
            per_frame_correct = ((fine_pred_binary == fine_label).float().mean(dim=-1) * event_mask.squeeze(-1) * coarse_mask)
            total_fine_correct += per_frame_correct.sum().item()
            total_fine_samples += (event_mask.squeeze(-1) * coarse_mask).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coarse_acc': f'{100.0 * total_coarse_correct / max(total_coarse_samples, 1):.2f}%',
            'fine_acc': f'{100.0 * total_fine_correct / max(total_fine_samples, 1):.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    coarse_acc = total_coarse_correct / max(total_coarse_samples, 1)
    fine_acc = total_fine_correct / max(total_fine_samples, 1)
    
    return avg_loss, coarse_acc, fine_acc


def validate_epoch(model, val_loader, device, fg_weight=5):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_coarse_correct = 0
    total_coarse_samples = 0
    total_fine_correct = 0
    total_fine_samples = 0
    
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch in pbar:
            frames = batch['frame'].to(device)
            coarse_label = batch['coarse_label'].to(device)
            fine_label = batch['fine_label'].to(device)
            coarse_mask = batch['coarse_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                coarse_pred, fine_pred = model(frames)
                
                # Coarse-grained loss
                coarse_loss = nn.functional.cross_entropy(
                    coarse_pred.reshape(-1, 2),
                    coarse_label.reshape(-1),
                    reduction='none'
                )
                coarse_loss = (coarse_loss * coarse_mask.reshape(-1)).mean()
                
                # Fine-grained loss - multi-label
                fine_loss = nn.functional.binary_cross_entropy_with_logits(
                    fine_pred,
                    fine_label.float(),
                    reduction='none'
                )
                event_mask = (coarse_label == 1).float().unsqueeze(-1)
                fine_loss = (fine_loss * event_mask * coarse_mask.unsqueeze(-1)).sum()
                fine_loss = fine_loss / (event_mask * coarse_mask.unsqueeze(-1)).sum().clamp(min=1)
                
                loss = coarse_loss + fg_weight * fine_loss
            
            total_loss += loss.item()
            
            # Statistics
            coarse_pred_labels = coarse_pred.argmax(dim=-1)
            coarse_correct = ((coarse_pred_labels == coarse_label) * coarse_mask).sum()
            total_coarse_correct += coarse_correct.item()
            total_coarse_samples += coarse_mask.sum().item()
            
            fine_pred_binary = (torch.sigmoid(fine_pred) > 0.5).float()
            event_mask = (coarse_label == 1).float().unsqueeze(-1)
            per_frame_correct = ((fine_pred_binary == fine_label).float().mean(dim=-1) * event_mask.squeeze(-1) * coarse_mask)
            total_fine_correct += per_frame_correct.sum().item()
            total_fine_samples += (event_mask.squeeze(-1) * coarse_mask).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'coarse_acc': f'{100.0 * total_coarse_correct / max(total_coarse_samples, 1):.2f}%',
                'fine_acc': f'{100.0 * total_fine_correct / max(total_fine_samples, 1):.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    coarse_acc = total_coarse_correct / max(total_coarse_samples, 1)
    fine_acc = total_fine_correct / max(total_fine_samples, 1)
    
    return avg_loss, coarse_acc, fine_acc


def main():
    parser = argparse.ArgumentParser(description='VTN Stage 1: Pre-training on F3Set')
    
    # Data paths
    parser.add_argument('--frame_dir', type=str, required=True,
                        help='Path to F3Set extracted frames')
    parser.add_argument('--dataset_name', type=str, default='f3set-tennis',
                        help='Dataset name (default: f3set-tennis)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    
    # Model parameters
    parser.add_argument('--vtn_spatial_size', type=str, default='small',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Size of ViT spatial backbone (default: small)')
    parser.add_argument('--vtn_temporal_type', type=str, default='longformer',
                        choices=['longformer', 'linformer', 'transformer'],
                        help='Type of temporal transformer (default: longformer)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained ViT weights')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    
    # Training parameters
    parser.add_argument('--clip_len', type=int, default=96,
                        help='Number of frames per clip (default: 96)')
    parser.add_argument('--crop_dim', type=int, default=224,
                        help='Crop dimension for images (default: 224)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Frame stride (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--fg_weight', type=float, default=5.0,
                        help='Weight for fine-grained loss (default: 5.0)')
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement (default: 0.001)')
    
    # Resume training
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VTN Stage 1: Pre-training on F3Set")
    print("=" * 80)
    print()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    store_json(config_path, vars(args), pretty=True)
    print(f"✓ Configuration saved to: {config_path}")
    
    # Load classes
    elements_file = os.path.join('F3Set', 'data', args.dataset_name, 'elements.txt')
    if not os.path.exists(elements_file):
        elements_file = os.path.join('MD-FED', 'data', args.dataset_name, 'elements.txt')
    
    print(f"\nLoading classes from: {elements_file}")
    classes = load_classes(elements_file)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes")
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Calculate dataset size (similar to MD-FED)
    epoch_num_frames = 500000
    dataset_len = epoch_num_frames // (args.clip_len * args.stride)
    print(f"Dataset size: {dataset_len} clips per epoch")
    
    train_json = os.path.join('F3Set', 'data', args.dataset_name, 'train.json')
    val_json = os.path.join('F3Set', 'data', args.dataset_name, 'val.json')
    
    if not os.path.exists(train_json):
        train_json = os.path.join('MD-FED', 'data', args.dataset_name, 'train.json')
        val_json = os.path.join('MD-FED', 'data', args.dataset_name, 'val.json')
    
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, args.clip_len, dataset_len,
        is_eval=False, stage=1,
        crop_dim=args.crop_dim, stride=args.stride
    )
    train_data.print_info()
    
    val_data = ActionSeqDataset(
        classes, val_json,
        args.frame_dir, args.clip_len, dataset_len // 4,
        is_eval=True, stage=1,
        crop_dim=args.crop_dim, stride=args.stride
    )
    val_data.print_info()
    
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    print("\nCreating VTN model...")
    print(f"  Spatial size: {args.vtn_spatial_size}")
    print(f"  Temporal type: {args.vtn_temporal_type}")
    print(f"  Clip length: {args.clip_len}")
    print(f"  Crop dimension: {args.crop_dim}")
    
    model = VTNModel(
        num_classes=num_classes,
        clip_len=args.clip_len,
        crop_dim=args.crop_dim,
        vtn_spatial_size=args.vtn_spatial_size,
        vtn_temporal_type=args.vtn_temporal_type,
        pretrained=not args.no_pretrained,
        pretrained_path=args.pretrained_path
    ).to(args.device)
    
    # Optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    
    if args.resume_checkpoint:
        print(f"\nResuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', [])
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_coarse_acc, train_fine_acc = train_epoch(
            model, train_loader, optimizer, scaler, args.device, args.fg_weight
        )
        
        # Validate
        val_loss, val_coarse_acc, val_fine_acc = validate_epoch(
            model, val_loader, args.device, args.fg_weight
        )
        
        # Log results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Coarse Acc: {100*train_coarse_acc:.2f}%, Fine Acc: {100*train_fine_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Coarse Acc: {100*val_coarse_acc:.2f}%, Fine Acc: {100*val_fine_acc:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_coarse_acc': train_coarse_acc,
            'train_fine_acc': train_fine_acc,
            'val_loss': val_loss,
            'val_coarse_acc': val_coarse_acc,
            'val_fine_acc': val_fine_acc
        })
        
        # Save best model
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_coarse_acc': val_coarse_acc,
                'val_fine_acc': val_fine_acc
            }, best_model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save last checkpoint (for resuming training)
        last_checkpoint_path = os.path.join(args.save_dir, 'last_checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'history': history
        }, last_checkpoint_path)
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Save history
        history_path = os.path.join(args.save_dir, 'history.json')
        store_json(history_path, history, pretty=True)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
