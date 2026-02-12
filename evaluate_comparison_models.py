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
            temporal_type=vtn_temporal_type,
            pretrained=False  # 不需要预训练权重，直接加载 checkpoint
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
    print(f"\nLoading trained checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if model_type == 'mdfed':
        model.load(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("✓ Loaded trained model successfully (checkpoint weights loaded)")
    
    return model


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
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save detailed predictions to .npz file for analysis'
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
    
    # Initialize prediction dictionaries for each video (像 MD-FED 一样)
    print(f"\n{'='*80}")
    print("Predicting on all rallies...")
    print(f"{'='*80}\n")
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, 2), np.float32),      # coarse scores
            np.zeros((video_len, len(classes)), np.float32),  # fine scores
            np.zeros(video_len, np.int32)              # support count
        )
    
    # Predict on all clips and accumulate
    from torch.utils.data import DataLoader
    batch_size = 1
    for clip in tqdm(DataLoader(dataset, num_workers=2, pin_memory=True, batch_size=batch_size),
                     desc="Predicting clips"):
        
        frames = clip['frame'].to(args.device)
        if len(frames.shape) == 6:
            frames = frames[:, 0]
        
        with torch.no_grad():
            if args.model_type in ['vtn', 'i3d']:
                coarse_pred, fine_pred = model(frames)
            else:  # mdfed
                flow = clip.get('flow')
                skeleton = clip.get('skeleton')
                if flow is not None:
                    flow = flow.to(args.device)
                    if len(flow.shape) == 6:
                        flow = flow[:, 0]
                if skeleton is not None:
                    skeleton = skeleton.to(args.device)
                    if len(skeleton.shape) == 6:
                        skeleton = skeleton[:, 0]
                coarse_pred, fine_pred = model(frames, flow, skeleton)
        
        # Process each video in the batch
        for i in range(frames.shape[0]):
            video = clip['video'][i]
            coarse_scores, fine_scores, support = pred_dict[video]
            
            # Get predictions for this clip
            coarse_pred_scores = torch.softmax(coarse_pred[i], dim=-1).cpu().numpy()
            fine_pred_scores = torch.sigmoid(fine_pred[i]).cpu().numpy()
            
            start = clip['start'][i].item()
            if start < 0:
                coarse_pred_scores = coarse_pred_scores[-start:, :]
                fine_pred_scores = fine_pred_scores[-start:, :]
                start = 0
            
            end = start + coarse_pred_scores.shape[0]
            if end >= coarse_scores.shape[0]:
                end = coarse_scores.shape[0]
                coarse_pred_scores = coarse_pred_scores[:end - start, :]
                fine_pred_scores = fine_pred_scores[:end - start, :]
            
            # Accumulate predictions
            coarse_scores[start:end, :] += coarse_pred_scores
            fine_scores[start:end, :] += fine_pred_scores
            support[start:end] += 1
    
    print("\nAveraging predictions...")
    
    # Average predictions and get labels
    all_coarse_pred = []
    all_fine_pred = []
    all_coarse_gt = []
    all_fine_gt = []
    
    for video in tqdm(sorted(pred_dict.keys()), desc="Processing videos"):
        coarse_scores, fine_scores, support = pred_dict[video]
        
        # Average by support count
        support_mask = support > 0
        coarse_scores[support_mask] /= support[support_mask, None]
        fine_scores[support_mask] /= support[support_mask, None]
        
        # Get predictions
        coarse_pred_labels = np.argmax(coarse_scores, axis=-1)
        fine_pred_labels = (fine_scores > 0.5).astype(int)
        
        # Get ground truth
        coarse_gt, fine_gt = dataset.get_labels(video)
        
        # Ensure same length
        min_len = min(len(coarse_gt), len(coarse_pred_labels))
        all_coarse_pred.append(coarse_pred_labels[:min_len])
        all_fine_pred.append(fine_pred_labels[:min_len])
        all_coarse_gt.append(coarse_gt[:min_len])
        all_fine_gt.append(fine_gt[:min_len])
    
    # Concatenate all
    all_coarse_pred = np.concatenate(all_coarse_pred)
    all_fine_pred = np.concatenate(all_fine_pred, axis=0)
    all_coarse_gt = np.concatenate(all_coarse_gt)
    all_fine_gt = np.concatenate(all_fine_gt, axis=0)
    
    print(f"\nTotal frames for evaluation: {len(all_coarse_gt):,}")
    
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
    
    # Show prediction vs ground truth statistics
    print(f"\n{'='*80}")
    print("Prediction Statistics")
    print(f"{'='*80}\n")
    
    # Coarse-grained (event detection) statistics
    print("Coarse-grained (Event Detection):")
    print(f"  Ground Truth: Event={np.sum(all_coarse_gt == 1):,}, Non-event={np.sum(all_coarse_gt == 0):,}")
    print(f"  Predicted:    Event={np.sum(all_coarse_pred == 1):,}, Non-event={np.sum(all_coarse_pred == 0):,}")
    print(f"  Correct:      {np.sum(all_coarse_pred == all_coarse_gt):,} / {len(all_coarse_gt):,} ({coarse_acc:.2%})")
    print()
    
    # Fine-grained (element) statistics per class
    print("Fine-grained (Elements) - Per Class:")
    classes_inv = {v-1: k for k, v in classes.items()}  # Convert to 0-indexed
    print(f"{'Class':<20} {'GT Total':>10} {'Pred Total':>10} {'Correct':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 80)
    
    for class_idx in range(len(classes)):
        class_name = classes_inv.get(class_idx, f"class_{class_idx}")
        gt_count = np.sum(all_fine_gt[:, class_idx] == 1)
        pred_count = np.sum(all_fine_pred[:, class_idx] == 1)
        correct_count = np.sum((all_fine_pred[:, class_idx] == 1) & (all_fine_gt[:, class_idx] == 1))
        
        class_precision = correct_count / pred_count if pred_count > 0 else 0
        class_recall = correct_count / gt_count if gt_count > 0 else 0
        
        print(f"{class_name:<20} {gt_count:>10,} {pred_count:>10,} {correct_count:>10,} {class_precision:>10.2%} {class_recall:>10.2%}")
    
    print()
    
    # Sample predictions
    print(f"\n{'='*80}")
    print("Sample Predictions (first 20 event frames)")
    print(f"{'='*80}\n")
    
    # Find event frames
    event_indices = np.where(all_coarse_gt == 1)[0]
    sample_indices = event_indices[:20] if len(event_indices) > 20 else event_indices
    
    if len(sample_indices) > 0:
        print(f"{'Frame':>6} | {'GT Event':>9} | {'Pred Event':>11} | GT Elements → Pred Elements")
        print("-" * 100)
        
        for idx in sample_indices:
            gt_event = "Event" if all_coarse_gt[idx] == 1 else "Non-event"
            pred_event = "Event" if all_coarse_pred[idx] == 1 else "Non-event"
            
            # Get element labels
            gt_elements = [classes_inv.get(i, f"c{i}") for i in range(len(classes)) if all_fine_gt[idx, i] == 1]
            pred_elements = [classes_inv.get(i, f"c{i}") for i in range(len(classes)) if all_fine_pred[idx, i] == 1]
            
            gt_str = ", ".join(gt_elements) if gt_elements else "None"
            pred_str = ", ".join(pred_elements) if pred_elements else "None"
            
            match_symbol = "✓" if gt_event == pred_event else "✗"
            
            print(f"{idx:>6} | {gt_event:>9} | {pred_event:>11} {match_symbol} | {gt_str} → {pred_str}")
    else:
        print("No event frames found in ground truth.")
    
    print()
    
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
        'fine_f1': float(f1),
        'coarse_stats': {
            'gt_events': int(np.sum(all_coarse_gt == 1)),
            'gt_non_events': int(np.sum(all_coarse_gt == 0)),
            'pred_events': int(np.sum(all_coarse_pred == 1)),
            'pred_non_events': int(np.sum(all_coarse_pred == 0))
        }
    }
    
    store_json(args.output_file, output_data, pretty=True)
    print(f"✓ Results saved to: {args.output_file}")
    
    # Save detailed predictions if requested
    if args.save_predictions:
        predictions_file = args.output_file.replace('.json', '_predictions.npz')
        np.savez_compressed(
            predictions_file,
            coarse_gt=all_coarse_gt,
            coarse_pred=all_coarse_pred,
            fine_gt=all_fine_gt,
            fine_pred=all_fine_pred
        )
        print(f"✓ Detailed predictions saved to: {predictions_file}")
    
    print()


if __name__ == '__main__':
    main()
