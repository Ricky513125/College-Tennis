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
from train_tsm_comparison import TSM_Flow_MD_FED as TSM_Model
from train_stgcn_comparison import STGCN_MD_FED as STGCN_Model
from train_rgb_flow_fusion import RGB_Flow_Fusion_MD_FED as RGB_Flow_Model
from train_rgb_flow_skeleton_fusion import RGB_Flow_Skeleton_Fusion_MD_FED as RGB_Flow_Skeleton_Model
from util.dataset import load_classes
from util.io import load_json, store_json
from dataset.input_process import ActionSeqVideoDataset
from util.eval import edit_score


def load_model(model_type, checkpoint_path, num_classes, device='cuda', 
               vtn_spatial_size='small', vtn_temporal_type='longformer',
               clip_len=96, crop_dim=224, tsm_visual_arch='rny002_tsm', tsm_temporal_arch='gru',
               stgcn_skeleton_arch='stgcn++', stgcn_temporal_arch='gru',
               rgb_flow_visual_arch='rny002_tsm', rgb_flow_temporal_arch='gru', rgb_flow_fusion_method='add',
               rgb_flow_skeleton_visual_arch='rny002_tsm', rgb_flow_skeleton_skeleton_arch='stgcn++',
               rgb_flow_skeleton_temporal_arch='gru', rgb_flow_skeleton_fusion_method='concat'):
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
    elif model_type == 'tsm':
        print(f"  Visual arch: {tsm_visual_arch}")
        print(f"  Temporal arch: {tsm_temporal_arch}")
        print(f"  Clip len: {clip_len}")
        print(f"  Crop dim: {crop_dim}")
        
        model = TSM_Model(
            num_classes=num_classes,
            clip_len=clip_len,
            visual_arch=tsm_visual_arch,
            temporal_arch=tsm_temporal_arch,
            pretrained=False  # 不需要预训练权重，直接加载 checkpoint
        )
    elif model_type == 'stgcn':
        print(f"  Skeleton arch: {stgcn_skeleton_arch}")
        print(f"  Temporal arch: {stgcn_temporal_arch}")
        print(f"  Clip len: {clip_len}")
        
        model = STGCN_Model(
            num_classes=num_classes,
            clip_len=clip_len,
            skeleton_arch=stgcn_skeleton_arch,
            temporal_arch=stgcn_temporal_arch
        )
    elif model_type == 'rgb_flow':
        print(f"  Visual arch: {rgb_flow_visual_arch}")
        print(f"  Temporal arch: {rgb_flow_temporal_arch}")
        print(f"  Fusion method: {rgb_flow_fusion_method}")
        print(f"  Clip len: {clip_len}")
        print(f"  Crop dim: {crop_dim}")
        
        model = RGB_Flow_Model(
            num_classes=num_classes,
            clip_len=clip_len,
            visual_arch=rgb_flow_visual_arch,
            temporal_arch=rgb_flow_temporal_arch,
            fusion_method=rgb_flow_fusion_method,
            pretrained=False  # 不需要预训练权重，直接加载 checkpoint
        )
    elif model_type == 'rgb_flow_skeleton_fusion':
        print(f"  Visual arch: {rgb_flow_skeleton_visual_arch}")
        print(f"  Skeleton arch: {rgb_flow_skeleton_skeleton_arch}")
        print(f"  Temporal arch: {rgb_flow_skeleton_temporal_arch}")
        print(f"  Fusion method: {rgb_flow_skeleton_fusion_method}")
        print(f"  Clip len: {clip_len}")
        print(f"  Crop dim: {crop_dim}")
        
        model = RGB_Flow_Skeleton_Model(
            num_classes=num_classes,
            clip_len=clip_len,
            visual_arch=rgb_flow_skeleton_visual_arch,
            skeleton_arch=rgb_flow_skeleton_skeleton_arch,
            temporal_arch=rgb_flow_skeleton_temporal_arch,
            fusion_method=rgb_flow_skeleton_fusion_method,
            pretrained=False  # 不需要预训练权重，直接加载 checkpoint
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if model_type == 'mdfed':
        model.load(checkpoint)
    else:
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Checkpoint format: {'epoch': ..., 'model_state_dict': {...}, ...}
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Alternative format: {'state_dict': {...}, ...}
            state_dict = checkpoint['state_dict']
        else:
            # Direct state_dict format
            state_dict = checkpoint
        
        # Filter out BatchNorm layers with dimension mismatch (especially for STGCN)
        # This can happen if training and evaluation use different graph configurations
        filtered_state_dict = {}
        model_state_dict = model.state_dict()
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                model_value = model_state_dict[key]
                # Check if shapes match
                if value.shape == model_value.shape:
                    filtered_state_dict[key] = value
                else:
                    # Skip BatchNorm layers with dimension mismatch
                    if 'running_mean' in key or 'running_var' in key or 'weight' in key or 'bias' in key:
                        if 'data_bn' in key or 'bn' in key:
                            print(f"⚠️  Skipping {key} due to shape mismatch: checkpoint {value.shape} vs model {model_value.shape}")
                            continue
                    # For other layers, still try to load if possible
                    filtered_state_dict[key] = value
        
        # Load with strict=False to handle missing/unexpected keys gracefully
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    - {key}")
            else:
                print(f"    (showing first 10 of {len(missing_keys)} missing keys)")
                for key in missing_keys[:10]:
                    print(f"    - {key}")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                print(f"    (showing first 10 of {len(unexpected_keys)} unexpected keys)")
                for key in unexpected_keys[:10]:
                    print(f"    - {key}")
    
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
        choices=['vtn', 'i3d', 'mdfed', 'tsm', 'stgcn', 'rgb_flow', 'rgb_flow_skeleton_fusion'],
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
        help='Path to optical flow directory (required for MD-FED and TSM)'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        default=None,
        help='Path to skeleton (pose) directory (required for MD-FED and STGCN)'
    )
    parser.add_argument(
        '--elements_file',
        type=str,
        default=None,
        help='Path to elements.txt file (default: MD-FED/data/f3set-tennis-sub/elements.txt or checkpoint_dir/elements.txt)'
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
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for evaluation (default: 1)'
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
    
    # TSM-specific parameters
    parser.add_argument(
        '--tsm_visual_arch',
        type=str,
        default='rny002_tsm',
        choices=['rny002_tsm', 'rn50_tsm'],
        help='TSM visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--tsm_temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru'],
        help='TSM temporal architecture (default: gru)'
    )
    
    # STGCN-specific parameters
    parser.add_argument(
        '--stgcn_skeleton_arch',
        type=str,
        default='stgcn++',
        choices=['stgcn++', 'stgcn'],
        help='STGCN skeleton architecture (default: stgcn++)'
    )
    parser.add_argument(
        '--stgcn_temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru'],
        help='STGCN temporal architecture (default: gru)'
    )
    
    # RGB+Flow-specific parameters
    parser.add_argument(
        '--rgb_flow_visual_arch',
        type=str,
        default='rny002_tsm',
        choices=['rny002_tsm', 'rny002', 'rn50_tsm', 'rn50'],
        help='RGB+Flow visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--rgb_flow_temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru'],
        help='RGB+Flow temporal architecture (default: gru)'
    )
    parser.add_argument(
        '--rgb_flow_fusion_method',
        type=str,
        default='add',
        choices=['add', 'concat', 'weighted'],
        help='RGB+Flow fusion method (default: add)'
    )
    
    # RGB+Flow+Skeleton Fusion-specific parameters
    parser.add_argument(
        '--rgb_flow_skeleton_visual_arch',
        type=str,
        default='rny002_tsm',
        choices=['rny002_tsm', 'rn50_tsm'],
        help='RGB+Flow+Skeleton visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--rgb_flow_skeleton_skeleton_arch',
        type=str,
        default='stgcn++',
        choices=['stgcn++', 'stgcn'],
        help='RGB+Flow+Skeleton skeleton architecture (default: stgcn++)'
    )
    parser.add_argument(
        '--rgb_flow_skeleton_temporal_arch',
        type=str,
        default='gru',
        choices=['gru', 'deeper_gru'],
        help='RGB+Flow+Skeleton temporal architecture (default: gru)'
    )
    parser.add_argument(
        '--rgb_flow_skeleton_fusion_method',
        type=str,
        default='concat',
        choices=['add', 'concat', 'weighted'],
        help='RGB+Flow+Skeleton fusion method (default: concat)'
    )
    
    args = parser.parse_args()
    
    # Set default output file
    if args.output_file is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_file = os.path.join(checkpoint_dir, 'evaluation_results.json')
    
    # Set default elements_file if not provided
    if args.elements_file is None:
        # Try checkpoint directory first
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_elements = os.path.join(checkpoint_dir, 'elements.txt')
        if os.path.exists(checkpoint_elements):
            args.elements_file = checkpoint_elements
        else:
            # Try MD-FED data directory
            md_fed_elements = os.path.join(Path(__file__).parent, 'MD-FED', 'data', 'f3set-tennis-sub', 'elements.txt')
            if os.path.exists(md_fed_elements):
                args.elements_file = md_fed_elements
            else:
                # Try current directory
                current_elements = 'elements.txt'
                if os.path.exists(current_elements):
                    args.elements_file = current_elements
                else:
                    raise FileNotFoundError(
                        f"Could not find elements.txt. Please specify --elements_file or ensure it exists in:\n"
                        f"  - {checkpoint_elements}\n"
                        f"  - {md_fed_elements}\n"
                        f"  - {current_elements}"
                    )
    
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
        crop_dim=args.crop_dim,
        tsm_visual_arch=args.tsm_visual_arch,
        tsm_temporal_arch=args.tsm_temporal_arch,
        stgcn_skeleton_arch=args.stgcn_skeleton_arch,
        stgcn_temporal_arch=args.stgcn_temporal_arch,
        rgb_flow_visual_arch=args.rgb_flow_visual_arch,
        rgb_flow_temporal_arch=args.rgb_flow_temporal_arch,
        rgb_flow_fusion_method=args.rgb_flow_fusion_method,
        rgb_flow_skeleton_visual_arch=args.rgb_flow_skeleton_visual_arch,
        rgb_flow_skeleton_skeleton_arch=args.rgb_flow_skeleton_skeleton_arch,
        rgb_flow_skeleton_temporal_arch=args.rgb_flow_skeleton_temporal_arch,
        rgb_flow_skeleton_fusion_method=args.rgb_flow_skeleton_fusion_method
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
        flow_dir=args.flow_dir if args.model_type in ['mdfed', 'tsm', 'rgb_flow', 'rgb_flow_skeleton_fusion'] else None,
        pose_dir=args.pose_dir if args.model_type in ['mdfed', 'stgcn', 'rgb_flow_skeleton_fusion'] else None,
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
    for clip in tqdm(DataLoader(dataset, num_workers=2, pin_memory=True, batch_size=args.batch_size),
                     desc="Predicting clips"):
        
        frames = clip['frame'].to(args.device)
        if len(frames.shape) == 6:
            frames = frames[:, 0]
        
        with torch.no_grad():
            if args.model_type in ['vtn', 'i3d']:
                coarse_pred, fine_pred = model(frames)
            elif args.model_type == 'tsm':
                # TSM only uses flow
                flow = clip.get('flow')
                if flow is None:
                    raise ValueError("TSM model requires flow data, but flow_dir not provided")
                flow = flow.to(args.device)
                if len(flow.shape) == 6:
                    flow = flow[:, 0]
                coarse_pred, fine_pred = model(frames=None, flow=flow)
            elif args.model_type == 'stgcn':
                # STGCN only uses skeleton
                skeleton = clip.get('skeleton')
                if skeleton is None:
                    raise ValueError("STGCN model requires skeleton data, but pose_dir not provided")
                skeleton = skeleton.to(args.device)
                if len(skeleton.shape) == 6:
                    skeleton = skeleton[:, 0]
                coarse_pred, fine_pred = model(frames=None, flow=None, skeleton=skeleton)
            elif args.model_type == 'rgb_flow':
                # RGB + Flow fusion uses both RGB and flow
                flow = clip.get('flow')
                if flow is None:
                    raise ValueError("RGB+Flow model requires flow data, but flow_dir not provided")
                flow = flow.to(args.device)
                if len(flow.shape) == 6:
                    flow = flow[:, 0]
                coarse_pred, fine_pred = model(frames=frames, flow=flow)
            elif args.model_type == 'rgb_flow_skeleton_fusion':
                # RGB + Flow + Skeleton fusion uses all three modalities
                flow = clip.get('flow')
                skeleton = clip.get('skeleton')
                if flow is None:
                    raise ValueError("RGB+Flow+Skeleton model requires flow data, but flow_dir not provided")
                if skeleton is None:
                    raise ValueError("RGB+Flow+Skeleton model requires skeleton data, but pose_dir not provided")
                flow = flow.to(args.device)
                skeleton = skeleton.to(args.device)
                if len(flow.shape) == 6:
                    flow = flow[:, 0]
                if len(skeleton.shape) == 6:
                    skeleton = skeleton[:, 0]
                coarse_pred, fine_pred = model(frames=frames, flow=flow, skeleton=skeleton)
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
    
    print("\nAveraging predictions and calculating metrics...")
    
    # Process fine-grained predictions using dataset-specific rules (same as MD-FED)
    # This is critical for consistent evaluation
    dataset_name = 'ncaa-rally'  # or infer from annotations
    
    # Calculate metrics using same method as MD-FED (delta=1 for LCL)
    print(f"\n{'='*80}")
    print("Calculating metrics (using MD-FED evaluation method)...")
    print(f"{'='*80}\n")
    
    delta = 1  # Same as MD-FED default
    from itertools import groupby
    
    # Initialize F1 counters
    f1_lcl = np.zeros((1, 3), int)  # [tp, fp, fn]
    f1_element = np.zeros((len(classes), 3), int)  # [tp, fp, fn] per class
    f1_event = dict()  # {event_id: [tp, fp, fn]}
    edit_scores = []
    
    # Also collect all predictions for display
    all_coarse_pred_list = []
    all_fine_pred_list = []
    all_coarse_gt_list = []
    all_fine_gt_list = []
    
    # Process each video separately for proper sequence-level metrics
    for video in sorted(pred_dict.keys()):
        coarse_gt, fine_gt = dataset.get_labels(video)
        coarse_scores, fine_scores, support = pred_dict[video]
        support_mask = support > 0
        fine_scores[support_mask] /= support[support_mask, None]
        
        coarse_pred_labels = np.argmax(coarse_scores, axis=-1)
        fine_pred = np.zeros_like(fine_scores, int)
        
        # Apply dataset-specific rules
        if 'f3set-tennis-sub' in dataset_name or 'ncaa-rally' in dataset_name:
            for i in range(len(fine_scores)):
                for start, end in [[0, 2], [2, 5]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1
                if fine_scores[i, 13] > 0.5:
                    fine_pred[i, 13] = 1
                if fine_pred[i, 2] != 1:
                    for start, end in [[5, 7], [7, 13]]:
                        max_idx = np.argmax(fine_scores[i, start:end])
                        fine_pred[i, start + max_idx] = 1
        
        fine_pred = coarse_pred_labels[:, np.newaxis] * fine_pred
        
        # Ensure same length
        min_len = min(len(coarse_gt), len(coarse_pred_labels))
        coarse_pred = coarse_pred_labels[:min_len]
        fine_pred = fine_pred[:min_len]
        coarse_gt = coarse_gt[:min_len]
        fine_gt = fine_gt[:min_len]
        
        # F1 (LCL) - Event localization
        for i in range(len(coarse_pred)):
            if coarse_pred[i] == 1 and sum(coarse_gt[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 1:
                f1_lcl[0, 0] += 1  # tp
            if coarse_pred[i] == 1 and sum(coarse_gt[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 1] += 1  # fp
            if coarse_gt[i] == 1 and sum(coarse_pred[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 2] += 1  # fn
        
        # F1 (element) - Element-level
        for i in range(len(fine_pred)):
            for j in range(len(fine_pred[0])):
                if fine_pred[i, j] == 1 and sum(fine_gt[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
                    f1_element[j, 0] += 1  # tp
                if fine_pred[i, j] == 1 and sum(fine_gt[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 1] += 1  # fp
                if fine_gt[i, j] == 1 and sum(fine_pred[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 2] += 1  # fn
        
        # F1 (event) - Event-level
        labels = [int(''.join(str(x) for x in row), 2) for row in fine_gt]
        preds = [int(''.join(str(x) for x in row), 2) for row in fine_pred]
        preds = coarse_pred * preds
        
        for i in range(len(preds)):
            if preds[i] > 0 and preds[i] in labels[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [1, 0, 0]
                else:
                    f1_event[preds[i]][0] += 1
            if preds[i] > 0 and sum(labels[max(0, i - delta):min(len(preds), i + delta + 1)]) == 0:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [0, 1, 0]
                else:
                    f1_event[preds[i]][1] += 1
            if labels[i] > 0 and labels[i] not in preds[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if labels[i] not in f1_event:
                    f1_event[labels[i]] = [0, 0, 1]
                else:
                    f1_event[labels[i]][2] += 1
        
        # Edit Score
        gt = [k for k, g in groupby(labels) if k != 0]
        pred = [k for k, g in groupby(preds) if k != 0]
        edit_scores.append(edit_score(pred, gt))
        
        # Collect for display
        all_coarse_pred_list.append(coarse_pred)
        all_fine_pred_list.append(fine_pred)
        all_coarse_gt_list.append(coarse_gt)
        all_fine_gt_list.append(fine_gt)
    
    # Concatenate all for display
    all_coarse_pred = np.concatenate(all_coarse_pred_list)
    all_fine_pred = np.concatenate(all_fine_pred_list, axis=0)
    all_coarse_gt = np.concatenate(all_coarse_gt_list)
    all_fine_gt = np.concatenate(all_fine_gt_list, axis=0)
    
    # Calculate Mean F1 scores
    # Mean F1 (LCL)
    precision_lcl = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 1] + 1e-10)
    recall_lcl = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 2] + 1e-10)
    f1_lcl_mean = np.mean(2 * precision_lcl * recall_lcl / (precision_lcl + recall_lcl + 1e-10))
    
    # Mean F1 (event)
    f1_event_mean = 0
    count = 0
    for value in f1_event.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1_event_mean += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1_event_mean = f1_event_mean / count if count > 0 else 0
    
    # Mean F1 (element)
    precision_element = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 1] + 1e-10)
    recall_element = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 2] + 1e-10)
    f1_element_mean = np.mean(2 * precision_element * recall_element / (precision_element + recall_element + 1e-10))
    
    # Edit Score
    edit = sum(edit_scores) / len(edit_scores) if len(edit_scores) > 0 else 0
    
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
    print("Evaluation Results (MD-FED Metrics)")
    print(f"{'='*80}")
    print(f"Total frames: {len(all_coarse_gt):,}")
    print(f"Mean F1 (LCL): {f1_lcl_mean:.4f}")
    print(f"Mean F1 (event): {f1_event_mean:.4f}")
    print(f"Mean F1 (element): {f1_element_mean:.4f}")
    print(f"Edit Score: {edit:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    output_data = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'num_rallies': len(annotations),
        'num_frames': int(len(all_coarse_gt)),
        'mean_f1_lcl': float(f1_lcl_mean),
        'mean_f1_event': float(f1_event_mean),
        'mean_f1_element': float(f1_element_mean),
        'edit_score': float(edit),
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
