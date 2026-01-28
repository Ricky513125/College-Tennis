#!/usr/bin/env python3
"""
Test Stage 2 trained model on manually verified labeled campus tennis data.
This script can also perform few-shot learning (Stage 3 fine-tuning) if needed.
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

# Import train_MD-FED.py using importlib (handles hyphen in filename)
import importlib.util
train_md_fed_path = os.path.join(md_fed_dir, 'train_MD-FED.py')
spec = importlib.util.spec_from_file_location("train_MD_FED", train_md_fed_path)
train_MD_FED = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_MD_FED)

# Import from the module
MD_FED = train_MD_FED.MD_FED
evaluate = train_MD_FED.evaluate
get_best_epoch_and_history = train_MD_FED.get_best_epoch_and_history

from dataset.input_process import ActionSeqVideoDataset
from util.dataset import load_classes
from util.io import load_json, store_json


def convert_manual_annotations_to_md_fed_format(manual_annotations_file, output_file, elements_file):
    """
    Convert manual_annotations.json to MD-FED format.
    
    Args:
        manual_annotations_file: Path to manual_annotations.json
        output_file: Path to save converted JSON
        elements_file: Path to elements.txt to get class mapping
    """
    print(f"Converting manual annotations from {manual_annotations_file}...")
    
    # Load manual annotations
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)
    
    # Load class mapping
    classes = load_classes(elements_file)
    classes_inv = {v: k for k, v in classes.items()}
    
    converted_data = []
    
    for video_data in manual_data:
        # Extract video info
        video_name = video_data['video']
        num_frames = video_data['num_frames']
        fps = video_data.get('fps', 30.0)
        height = video_data.get('height', 1080)
        width = video_data.get('width', 1920)
        
        # Convert events to MD-FED format
        events = []
        for event in video_data.get('events', []):
            frame = event['frame']
            label_str = event['label']
            
            # Parse label string (e.g., "far_middle_serve_-_-_W_-_in")
            # Format: {near/far}_{position}_{action}_{hand}_{stroke}_{direction}_{outcome}
            parts = label_str.split('_')
            
            # Create label dict
            event_dict = {
                'frame': frame,
                'label': label_str,
            }
            
            # Add outcome if present
            if 'outcome' in event:
                event_dict['outcome'] = event['outcome']
            
            events.append(event_dict)
        
        # Create converted entry
        converted_entry = {
            'fps': fps,
            'height': height,
            'width': width,
            'num_frames': num_frames,
            'video': video_name,
            'far_name': video_data.get('far_name', 'Unknown'),
            'far_hand': video_data.get('far_hand', 'RH'),
            'far_set': video_data.get('far_set', 0),
            'far_game': video_data.get('far_game', 0),
            'far_point': video_data.get('far_point', 0),
            'near_name': video_data.get('near_name', 'Unknown'),
            'near_hand': video_data.get('near_hand', 'RH'),
            'near_set': video_data.get('near_set', 0),
            'near_game': video_data.get('near_game', 0),
            'near_point': video_data.get('near_point', 0),
            'events': events
        }
        
        converted_data.append(converted_entry)
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} videos to {output_file}")
    return converted_data


def load_stage2_model(checkpoint_dir, config_file=None, epoch=None, device='cuda'):
    """
    Load Stage 2 trained model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        config_file: Path to config.json (optional, will try to find it)
        epoch: Specific epoch to load (optional, will use best epoch if not specified)
        device: Device to load model on
    """
    # Load config
    if config_file is None:
        config_file = os.path.join(checkpoint_dir, 'config.json')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config = load_json(config_file)
    print(f"Loaded config from {config_file}")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Visual arch: {config['visual_arch']}")
    print(f"  Skeleton arch: {config['skeleton_arch']}")
    print(f"  Temporal arch: {config['temporal_arch']}")
    print(f"  Clip len: {config['clip_len']}")
    print(f"  Window: {config['window']}")
    print(f"  Stage: {config['stage']}")
    
    # Determine which epoch to load
    if epoch is None:
        # Get best epoch from loss.json
        loss_file = os.path.join(checkpoint_dir, 'loss.json')
        if os.path.exists(loss_file):
            losses = load_json(loss_file)
            # Find best epoch based on val_edit score
            best_epoch = max(losses, key=lambda x: x.get('val_edit', 0))['epoch']
            print(f"Best epoch from loss.json: {best_epoch}")
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            epochs = [int(f.replace('checkpoint_', '').replace('.pt', '')) for f in checkpoints]
            best_epoch = max(epochs)
            print(f"Using latest checkpoint: epoch {best_epoch}")
    else:
        best_epoch = epoch
        print(f"Loading specified epoch: {best_epoch}")
    
    # Load classes
    elements_file = os.path.join('MD-FED', 'data', config['dataset'], 'elements.txt')
    if not os.path.exists(elements_file):
        # Try alternative location
        elements_file = os.path.join('data', config['dataset'], 'elements.txt')
    if not os.path.exists(elements_file):
        raise FileNotFoundError(f"Elements file not found. Tried: {elements_file}")
    
    classes = load_classes(elements_file)
    print(f"Loaded {len(classes)} classes from {elements_file}")
    
    # Create model
    model = MD_FED(
        len(classes),
        config['visual_arch'],
        config['skeleton_arch'],
        config['temporal_arch'],
        clip_len=config['clip_len'],
        step=config.get('stride', 2),
        window=config['window'],
        stage=2,  # Stage 2 model
        multi_gpu=False
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{best_epoch:03d}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load(checkpoint)
    model._model.eval()
    
    print(f"✓ Model loaded successfully (epoch {best_epoch})")
    
    return model, classes, config


def test_model_on_manual_data(
    model,
    classes,
    manual_annotations_file,
    frame_dir,
    flow_dir=None,
    pose_dir=None,
    config=None,
    output_dir=None
):
    """
    Test model on manually verified data.
    
    Args:
        model: Loaded MD_FED model
        classes: Class dictionary
        manual_annotations_file: Path to manual annotations JSON
        frame_dir: Directory containing extracted frames
        flow_dir: Directory containing optical flow (optional)
        pose_dir: Directory containing pose data (optional)
        config: Model config dictionary
        output_dir: Directory to save results (optional)
    """
    print(f"\n{'='*60}")
    print("Testing Model on Manually Verified Data")
    print(f"{'='*60}\n")
    
    # Convert manual annotations to MD-FED format
    temp_json = os.path.join(output_dir or '.', 'temp_manual_test.json')
    elements_file = os.path.join('MD-FED', 'data', config['dataset'], 'elements.txt')
    if not os.path.exists(elements_file):
        elements_file = os.path.join('data', config['dataset'], 'elements.txt')
    
    converted_data = convert_manual_annotations_to_md_fed_format(
        manual_annotations_file, temp_json, elements_file
    )
    
    # Create dataset
    print(f"\nCreating dataset from {temp_json}...")
    print(f"  Frame dir: {frame_dir}")
    if flow_dir:
        print(f"  Flow dir: {flow_dir}")
    if pose_dir:
        print(f"  Pose dir: {pose_dir}")
    
    test_dataset = ActionSeqVideoDataset(
        classes,
        temp_json,
        frame_dir,
        config['clip_len'],
        overlap_len=config['clip_len'] // 2,
        crop_dim=config.get('crop_dim', 224),
        stride=config.get('stride', 2),
        flow_dir=flow_dir,
        pose_dir=pose_dir,
        is_test=True
    )
    
    test_dataset.print_info()
    
    # Evaluate
    print(f"\n{'='*60}")
    print("Running Evaluation")
    print(f"{'='*60}\n")
    
    edit_score = evaluate(
        model,
        test_dataset,
        classes,
        delta=10,
        window=config['window'],
        dataset_name=config['dataset']
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"  Edit Score: {edit_score:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results = {
            'edit_score': edit_score,
            'num_videos': len(converted_data),
            'dataset': config['dataset'],
            'checkpoint_dir': config.get('checkpoint_dir', 'unknown')
        }
        results_file = os.path.join(output_dir, 'test_results.json')
        store_json(results_file, results, pretty=True)
        print(f"Results saved to {results_file}")
    
    # Cleanup temp file
    if os.path.exists(temp_json):
        os.remove(temp_json)
    
    return edit_score


def main():
    parser = argparse.ArgumentParser(
        description='Test Stage 2 model on manually verified labeled campus tennis data'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing Stage 2 checkpoints (e.g., ./MD-FED/md_fed_outputs/stage2)'
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
        help='Directory containing optical flow files (optional)'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        default=None,
        help='Directory containing pose/skeleton files (optional)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch to load (default: best epoch from loss.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save test results (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading Stage 2 model from {args.checkpoint_dir}...")
    model, classes, config = load_stage2_model(
        args.checkpoint_dir,
        epoch=args.epoch,
        device=args.device
    )
    
    # Store checkpoint dir in config for results
    config['checkpoint_dir'] = args.checkpoint_dir
    
    # Test on manual data
    edit_score = test_model_on_manual_data(
        model,
        classes,
        args.manual_annotations,
        args.frame_dir,
        flow_dir=args.flow_dir,
        pose_dir=args.pose_dir,
        config=config,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Testing complete! Final Edit Score: {edit_score:.4f}")


if __name__ == '__main__':
    main()
