#!/usr/bin/env python3
"""
通用评估脚本，支持 MD-FED、VTN 和 I3D 模型

Usage:
    # 评估 VTN
    python evaluate_comparison_models.py \\
        --model_type vtn \\
        --checkpoint ./vtn_outputs/best_model.pt \\
        --manual_annotations manual_annotations.json \\
        --frame_dir /path/to/frames

    # 评估 I3D
    python evaluate_comparison_models.py \\
        --model_type i3d \\
        --checkpoint ./i3d_outputs/best_model.pt \\
        --manual_annotations manual_annotations.json \\
        --frame_dir /path/to/frames

    # 评估 MD-FED
    python evaluate_comparison_models.py \\
        --model_type mdfed \\
        --checkpoint ./MD-FED/md_fed_outputs/stage3/best_model.pt \\
        --manual_annotations manual_annotations.json \\
        --frame_dir /path/to/frames \\
        --flow_dir /path/to/flows
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add MD-FED to path
sys.path.insert(0, str(Path(__file__).parent / 'MD-FED'))

# Import models
from train_vtn_comparison import VTN_MD_FED as VTN_Model
from train_i3d_comparison import I3D_MD_FED as I3D_Model
from util.dataset import load_classes
from util.io import load_json, store_json
from dataset.input_process import ActionSeqVideoDataset
from util.eval import edit_score


def load_model(model_type, checkpoint_path, num_classes, device='cuda', 
               vtn_spatial_size='small', vtn_temporal_type='longformer',
               clip_len=96, crop_dim=224):
    """
    加载指定类型的模型
    """
    print(f"\nLoading {model_type.upper()} model...")
    
    if model_type == 'vtn':
        print(f"  Spatial size: {vtn_spatial_size}")
        print(f"  Temporal type: {vtn_temporal_type}")
        print(f"  Clip len: {clip_len}")
        print(f"  Crop dim: {crop_dim}")
        
        model = VTN_Model(
            num_classes=num_classes,
            clip_len=clip_len,
            img_size=crop_dim,
            spatial_size=vtn_spatial_size,
            temporal_type=vtn_temporal_type
        )
    elif model_type == 'i3d':
        model = I3D_Model(
            num_classes=num_classes,
            clip_len=96
        )
    elif model_type == 'mdfed':
        # Import MD-FED model
        import importlib.util
        train_md_fed_path = Path(__file__).parent / 'MD-FED' / 'train_MD-FED.py'
        spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
        train_MD_FED = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_MD_FED)
        
        MD_FED = train_MD_FED.MD_FED
        model = MD_FED(
            num_classes,
            visual_arch='resnet',
            skeleton_arch='gcn',
            temporal_arch='gru',
            clip_len=96,
            stage=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if model_type == 'mdfed':
        model.load(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    return model


def predict_video(model, model_type, video_dataset, device='cuda'):
    """
    对一个视频进行预测
    """
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        video_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_coarse_preds = []
    all_fine_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frame'].to(device)
            
            # Handle different input shapes
            if len(frames.shape) == 6:  # [1, flip, clip_len, C, H, W]
                frames = frames[:, 0]  # Take first flip
            
            if model_type in ['vtn', 'i3d']:
                coarse_pred, fine_pred = model(frames)
            else:  # mdfed
                flow = batch.get('flow')
                skeleton = batch.get('skeleton')
                if flow is not None:
                    flow = flow.to(device)
                    if len(flow.shape) == 6:
                        flow = flow[:, 0]
                if skeleton is not None:
                    skeleton = skeleton.to(device)
                    if len(skeleton.shape) == 6:
                        skeleton = skeleton[:, 0]
                
                coarse_pred, fine_pred = model(frames, flow, skeleton)
            
            all_coarse_preds.append(coarse_pred.cpu())
            all_fine_preds.append(fine_pred.cpu())
    
    # Concatenate all predictions
    coarse_preds = torch.cat(all_coarse_preds, dim=1)  # [1, total_frames, 2]
    fine_preds = torch.cat(all_fine_preds, dim=1)  # [1, total_frames, num_classes]
    
    return coarse_preds[0], fine_preds[0]  # [total_frames, ...]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VTN, I3D, or MD-FED model on manual annotations'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['vtn', 'i3d', 'mdfed'],
        help='Model type to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
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
        help='Path to optical flow directory (only for MD-FED)'
    )
    parser.add_argument(
        '--elements_file',
        type=str,
        default='elements.txt',
        help='Path to elements.txt file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    # VTN-specific parameters
    parser.add_argument(
        '--vtn_spatial_size',
        type=str,
        default='small',
        choices=['tiny', 'small', 'base', 'large'],
        help='VTN spatial backbone size (default: small)'
    )
    parser.add_argument(
        '--vtn_temporal_type',
        type=str,
        default='longformer',
        choices=['longformer', 'linformer', 'transformer'],
        help='VTN temporal transformer type (default: longformer)'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='Number of frames per clip (default: 96)'
    )
    parser.add_argument(
        '--crop_dim',
        type=int,
        default=224,
        help='Crop dimension for images (default: 224)'
    )
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output_file is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_file = os.path.join(checkpoint_dir, 'evaluation_results.json')
    
    print("="*80)
    print(f"Evaluating {args.model_type.upper()} Model")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Annotations: {args.manual_annotations}")
    print(f"Frame dir: {args.frame_dir}")
    if args.flow_dir:
        print(f"Flow dir: {args.flow_dir}")
    print(f"Output: {args.output_file}")
    
    # Load classes
    print(f"\nLoading classes from: {args.elements_file}")
    classes = load_classes(args.elements_file)
    print(f"Loaded {len(classes)} classes")
    
    # Load annotations
    print(f"\nLoading annotations...")
    annotations = load_json(args.manual_annotations)
    print(f"Total rallies to evaluate: {len(annotations)}")
    
    # Load model
    model = load_model(
        args.model_type, args.checkpoint, len(classes), args.device,
        vtn_spatial_size=args.vtn_spatial_size,
        vtn_temporal_type=args.vtn_temporal_type,
        clip_len=args.clip_len,
        crop_dim=args.crop_dim
    )
    
    # Create dataset with all annotations
    print(f"\n{'='*80}")
    print("Creating dataset...")
    print(f"{'='*80}\n")
    
    dataset = ActionSeqVideoDataset(
        classes=classes,
        label_file=args.manual_annotations,
        frame_dir=args.frame_dir,
        clip_len=args.clip_len,
        overlap_len=args.clip_len // 2,
        crop_dim=args.crop_dim,
        stride=2,
        flow_dir=args.flow_dir if args.model_type == 'mdfed' else None,
        pose_dir=None,
        pad_len=0,
        flip=False,
        multi_crop=False,
        skip_partial_end=True
    )
    
    print(f"Dataset created with {len(dataset)} clips")
    
    # Predict on entire dataset
    print(f"\n{'='*80}")
    print("Predicting on all rallies...")
    print(f"{'='*80}\n")
    
    coarse_pred, fine_pred = predict_video(model, args.model_type, dataset, args.device)
    
    # Convert predictions
    coarse_pred_labels = torch.argmax(coarse_pred, dim=-1).numpy()
    fine_pred_labels = (torch.sigmoid(fine_pred) > 0.5).numpy()
    
    print(f"Predictions completed: {len(coarse_pred_labels):,} frames")
    
    # Get ground truth for all videos
    print(f"\n{'='*80}")
    print("Collecting ground truth labels...")
    print(f"{'='*80}\n")
    
    all_coarse_gt = []
    all_fine_gt = []
    
    video_names = [v[0] for v in dataset.videos]
    for video_name in tqdm(video_names, desc="Loading ground truth"):
        try:
            coarse_gt, fine_gt = dataset.get_labels(video_name)
            all_coarse_gt.append(coarse_gt)
            all_fine_gt.append(fine_gt)
        except Exception as e:
            print(f"\n❌ Error loading labels for {video_name}: {e}")
            continue
    
    # Concatenate all ground truth
    all_coarse_gt = np.concatenate(all_coarse_gt)
    all_fine_gt = np.concatenate(all_fine_gt, axis=0)
    
    # Ensure same length
    min_len = min(len(all_coarse_gt), len(coarse_pred_labels))
    all_coarse_gt = all_coarse_gt[:min_len]
    all_coarse_pred = coarse_pred_labels[:min_len]
    all_fine_gt = all_fine_gt[:min_len]
    all_fine_pred = fine_pred_labels[:min_len]
    
    print(f"Matched {min_len:,} frames for evaluation")
    
    # Calculate overall metrics
    print(f"\n{'='*80}")
    print("Calculating metrics...")
    print(f"{'='*80}\n")
    
    # Edit Score
    edit = edit_score(all_coarse_pred, all_coarse_gt, norm=True, bg_class=[0])
    
    # Coarse accuracy
    coarse_acc = np.mean(all_coarse_pred == all_coarse_gt)
    
    # Fine-grained metrics
    all_fine_pred_flat = all_fine_pred.reshape(-1)
    all_fine_gt_flat = all_fine_gt.reshape(-1)
    
    tp = np.sum((all_fine_pred_flat == 1) & (all_fine_gt_flat == 1))
    fp = np.sum((all_fine_pred_flat == 1) & (all_fine_gt_flat == 0))
    fn = np.sum((all_fine_pred_flat == 0) & (all_fine_gt_flat == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"Total frames: {len(all_coarse_gt):,}")
    print(f"Edit Score: {edit:.4f}")
    print(f"Coarse Accuracy: {coarse_acc:.4f}")
    print(f"Fine-grained Precision: {precision:.4f}")
    print(f"Fine-grained Recall: {recall:.4f}")
    print(f"Fine-grained F1: {f1:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    output_data = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'num_rallies': len(annotations),
        'num_frames': int(len(all_coarse_gt)),
        'edit_score': float(edit),
        'coarse_accuracy': float(coarse_acc),
        'fine_precision': float(precision),
        'fine_recall': float(recall),
        'fine_f1': float(f1)
    }
    
    store_json(args.output_file, output_data, pretty=True)
    print(f"✓ Results saved to: {args.output_file}\n")


if __name__ == '__main__':
    main()
