#!/usr/bin/env python3
"""
Generate annotations for videos using F3-set models.
This script runs inference on processed videos and generates annotation files.
"""

import os
import sys
import json
import argparse
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add F3Set to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F3Set'))

from dataset.frame import ActionSeqVideoDataset
from util.io import load_json, store_json
from util.dataset import load_classes


def get_best_epoch(model_dir, key='val_edit'):
    """Get the best epoch from loss.json"""
    loss_file = os.path.join(model_dir, 'loss.json')
    if os.path.isfile(loss_file):
        data = load_json(loss_file)
        best = max(data, key=lambda x: x[key])
        return best['epoch']
    return None


def get_last_epoch(model_dir):
    """Get the last checkpoint epoch"""
    regex = re.compile(r'checkpoint_(\d+)\.pt')
    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    if last_epoch < 0:
        raise ValueError(f"No checkpoint found in {model_dir}")
    return last_epoch


def load_model(model_dir, dataset_name='f3set-tennis', use_f3ed=False):
    """Load the F3-set model"""
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_json(config_path)
    
    # Determine which epoch to use
    if os.path.isfile(os.path.join(model_dir, 'loss.json')):
        best_epoch = get_best_epoch(model_dir)
        print(f'Using best epoch: {best_epoch}')
    else:
        best_epoch = get_last_epoch(model_dir)
        print(f'Using last epoch: {best_epoch}')
    
    # Override dataset if specified
    if dataset_name:
        config['dataset'] = dataset_name
    
    # For F3ED, use the num_classes from config (the model was trained with this)
    # For baseline, load from events.txt
    if use_f3ed:
        from train_f3set_f3ed import F3Set
        # Use num_classes from config (model was trained with this)
        num_classes = config.get('num_classes')
        if num_classes is None:
            # Fallback: load from elements.txt
            elements_file = os.path.join('F3Set', 'data', config['dataset'], 'elements.txt')
            if os.path.exists(elements_file):
                classes = load_classes(elements_file)
                num_classes = len(classes)
            else:
                raise ValueError(f"Cannot determine num_classes. Please check config.json or elements.txt")
        else:
            # Still load classes for mapping labels
            elements_file = os.path.join('F3Set', 'data', config['dataset'], 'elements.txt')
            if os.path.exists(elements_file):
                classes = load_classes(elements_file)
            else:
                # Create dummy classes mapping if file doesn't exist
                classes = {f'class_{i}': i+1 for i in range(num_classes)}
        
        model = F3Set(
            num_classes, 
            config['feature_arch'], 
            config['temporal_arch'], 
            clip_len=config['clip_len'],
            step=config['stride'], 
            window=config['window'], 
            use_ctx=config.get('use_ctx', False),
            multi_gpu=config.get('gpu_parallel', False)
        )
    else:
        from train_f3set_baselines import F3Set
        # Load classes from events.txt for baseline
        events_file = os.path.join('F3Set', 'data', config['dataset'], 'events.txt')
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"Events file not found: {events_file}")
        classes = load_classes(events_file)
        num_classes = len(classes) + 1  # +1 for background
        model = F3Set(
            num_classes, 
            config['feature_arch'], 
            config['temporal_arch'], 
            clip_len=config['clip_len'],
            step=config['stride'], 
            window=config['window'],
            multi_gpu=config.get('gpu_parallel', False)
        )
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, f'checkpoint_{best_epoch:03d}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    # F3ED checkpoints are saved as direct state_dict, not wrapped
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Standard wrapped format
            state_dict = checkpoint['model_state_dict']
        else:
            # Direct state_dict format (F3ED saves this way)
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load with strict=False to handle key mismatches
    # This allows loading even if some keys don't match exactly
    try:
        if isinstance(model._model, nn.DataParallel):
            model._model.module.load_state_dict(state_dict, strict=False)
        else:
            model._model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Some keys could not be loaded: {e}")
        print("Continuing with partial model loading...")
    
    # Set model to eval mode
    model._model.eval()
    
    if torch.cuda.is_available():
        model._model = model._model.cuda()
    
    return model, classes, config


def run_inference(model, dataset, classes, use_f3ed=False):
    """Run inference on the dataset using DataLoader"""
    from torch.utils.data import DataLoader
    
    INFERENCE_BATCH_SIZE = 4
    
    # Set model to eval mode
    model._model.eval()
    pred_dict = {}
    
    # Initialize prediction dictionaries
    for video, video_len, _ in dataset.videos:
        if use_f3ed:
            pred_dict[video] = (
                np.zeros((video_len, 2), np.float32),
                np.zeros((video_len, len(classes)), np.float32),
                np.zeros(video_len, np.int32)
            )
        else:
            pred_dict[video] = (
                np.zeros((video_len, len(classes) + 1), np.float32),
                np.zeros(video_len, np.int32)
            )
    
    # Run inference with DataLoader
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    with torch.no_grad():
        for clip in tqdm(DataLoader(
            dataset, 
            num_workers=4, 
            pin_memory=True,
            batch_size=batch_size
        ), desc="Running inference"):
            
            if batch_size > 1:
                if use_f3ed:
                    _, batch_coarse_scores, batch_fine_scores = model.predict(
                        clip['frame'], clip.get('hand', None)
                    )
                    for i in range(clip['frame'].shape[0]):
                        video = clip['video'][i]
                        coarse_scores, fine_scores, support = pred_dict[video]
                        # predict() already returns numpy arrays
                        coarse_pred_scores = batch_coarse_scores[i]
                        fine_pred_scores = batch_fine_scores[i]
                        
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
                        coarse_scores[start:end, :] += coarse_pred_scores
                        fine_scores[start:end, :] += fine_pred_scores
                        support[start:end] += 1
                else:
                    _, batch_pred_scores = model.predict(clip['frame'])
                    for i in range(clip['frame'].shape[0]):
                        video = clip['video'][i]
                        scores, support = pred_dict[video]
                        # predict() may return numpy arrays or tensors
                        if isinstance(batch_pred_scores[i], torch.Tensor):
                            pred_scores = batch_pred_scores[i].cpu().numpy()
                        else:
                            pred_scores = batch_pred_scores[i]
                        start = clip['start'][i].item()
                        if start < 0:
                            pred_scores = pred_scores[-start:, :]
                            start = 0
                        end = start + pred_scores.shape[0]
                        if end >= scores.shape[0]:
                            end = scores.shape[0]
                            pred_scores = pred_scores[:end - start, :]
                        scores[start:end, :] += pred_scores
                        support[start:end] += 1
    
    return pred_dict


def generate_annotations(metadata_file, frame_dir, model_dir, output_dir, 
                        dataset_name='f3set-tennis', use_f3ed=False):
    """
    Generate annotations for videos.
    
    Args:
        metadata_file: Path to video metadata JSON file
        frame_dir: Directory containing extracted frames
        model_dir: Directory containing trained model
        output_dir: Directory to save annotation outputs
        dataset_name: Name of the dataset (for loading classes)
        use_f3ed: Whether to use F3ED model instead of baseline
    """
    # Load model
    print("Loading model...")
    model, classes, config = load_model(model_dir, dataset_name, use_f3ed)
    
    # Load metadata
    print("Loading video metadata...")
    with open(metadata_file, 'r') as f:
        video_metadata = json.load(f)
    
    # Create dataset
    print("Creating dataset...")
    # Create a temporary JSON file for the dataset
    temp_json = os.path.join(output_dir, 'temp_videos.json')
    with open(temp_json, 'w') as f:
        json.dump(video_metadata, f)
    
    try:
        if use_f3ed:
            from dataset.frame_process import ActionSeqVideoDataset
            overlap_len = config['clip_len'] // 2
        else:
            from dataset.frame import ActionSeqVideoDataset
            overlap_len = 0
        
        # Check if frames exist before creating dataset
        print("Checking frame directories...")
        for video_entry in video_metadata[:3]:  # Check first 3 videos
            video_id = video_entry['video']
            video_frame_dir = os.path.join(frame_dir, video_id)
            if os.path.exists(video_frame_dir):
                frame_files = sorted([f for f in os.listdir(video_frame_dir) if f.endswith('.jpg')])
                print(f"  {video_id}: {len(frame_files)} frames found (expected {video_entry['num_frames']})")
                if len(frame_files) > 0:
                    print(f"    First frame: {frame_files[0]}, Last frame: {frame_files[-1]}")
            else:
                print(f"  WARNING: Frame directory not found: {video_frame_dir}")
        
        # Verify frame numbering matches expected format (000000.jpg, 000001.jpg, etc.)
        print("\nVerifying frame file format...")
        sample_video = video_metadata[0]['video']
        sample_frame_dir = os.path.join(frame_dir, sample_video)
        if os.path.exists(sample_frame_dir):
            frame_files = sorted([f for f in os.listdir(sample_frame_dir) if f.endswith('.jpg')])
            if frame_files:
                expected_first = '000000.jpg'
                if frame_files[0] != expected_first:
                    print(f"  WARNING: First frame is {frame_files[0]}, expected {expected_first}")
        
        dataset = ActionSeqVideoDataset(
            classes,
            temp_json,
            frame_dir,
            config['clip_len'],
            overlap_len=overlap_len,
            crop_dim=config['crop_dim'],
            stride=config['stride'],
            pad_len=0  # Set pad_len to 0 to avoid issues with negative indices
        )
        
        print(f"\nDataset created with {len(dataset)} clips")
        if len(dataset) == 0:
            raise ValueError("No clips created! Check frame directories and video metadata.")
        
        # Run inference
        print("Running inference...")
        pred_dict = run_inference(model, dataset, classes, use_f3ed)
        
        # Process predictions
        print("Processing predictions...")
        classes_inv = {v: k for k, v in classes.items()}
        classes_inv[0] = 'NA'
        
        # For F3ED, we need to load events.txt to match composite labels
        events_classes = None
        if use_f3ed:
            events_file = os.path.join('F3Set', 'data', dataset_name, 'events.txt')
            if os.path.exists(events_file):
                events_classes = load_classes(events_file)
                print(f"Loaded {len(events_classes)} event classes from {events_file}")
        
        def build_event_label_from_elements(fine_pred, classes_inv):
            """Build composite event label from fine-grained element predictions"""
            # Get active elements
            active_elements = []
            for idx in range(len(fine_pred)):
                if fine_pred[idx] == 1:
                    element_name = classes_inv[idx + 1]  # +1 because classes are 1-indexed
                    active_elements.append(element_name)
            
            if not active_elements:
                return None
            
            # Build label based on element categories
            # Format: {player}_{court_position}_{action_type}_{hand}_{stroke_type}_{direction}_{formation}_{outcome}
            player = None
            court_position = None
            action_type = None
            hand = None
            stroke_type = None
            direction = None
            formation = None
            outcome = None
            
            # Player: near (0) or far (1)
            if 'near' in active_elements:
                player = 'near'
            elif 'far' in active_elements:
                player = 'far'
            
            # Court position: deuce (2), middle (3), ad (4)
            if 'deuce' in active_elements:
                court_position = 'deuce'
            elif 'middle' in active_elements:
                court_position = 'middle'
            elif 'ad' in active_elements:
                court_position = 'ad'
            
            # Action type: serve (5), return (6), stroke (7)
            if 'serve' in active_elements:
                action_type = 'serve'
            elif 'return' in active_elements:
                action_type = 'return'
            elif 'stroke' in active_elements:
                action_type = 'stroke'
            
            # Hand: fh (8) or bh (9) - only for non-serve actions
            if action_type and action_type != 'serve':
                if 'fh' in active_elements:
                    hand = 'fh'
                elif 'bh' in active_elements:
                    hand = 'bh'
            
            # Stroke type: gs (10), slice (11), volley (12), smash (13), drop (14), lob (15)
            # Only for non-serve actions
            if action_type and action_type != 'serve':
                if 'gs' in active_elements:
                    stroke_type = 'gs'
                elif 'slice' in active_elements:
                    stroke_type = 'slice'
                elif 'volley' in active_elements:
                    stroke_type = 'volley'
                elif 'smash' in active_elements:
                    stroke_type = 'smash'
                elif 'drop' in active_elements:
                    stroke_type = 'drop'
                elif 'lob' in active_elements:
                    stroke_type = 'lob'
            
            # Direction: T (16), B (17), W (18), CC (19), DL (20), DM (21), II (22), IO (23)
            direction_candidates = ['T', 'B', 'W', 'CC', 'DL', 'DM', 'II', 'IO']
            for d in direction_candidates:
                if d in active_elements:
                    direction = d
                    break
            
            # Formation: approach (24)
            if 'approach' in active_elements:
                formation = 'approach'
            
            # Outcome: in (25), winner (26), forced-err (27), unforced-err (28)
            if 'in' in active_elements:
                outcome = 'in'
            elif 'winner' in active_elements:
                outcome = 'winner'
            elif 'forced-err' in active_elements:
                outcome = 'forced-err'
            elif 'unforced-err' in active_elements:
                outcome = 'unforced-err'
            
            # Build label string
            # Format: {player}_{court_position}_{action_type}_{hand}_{stroke_type}_{direction}_{formation}_{outcome}
            parts = [
                player or '-',
                court_position or '-',
                action_type or '-',
                hand or '-',
                stroke_type or '-',
                direction or '-',
                formation or '-',
                outcome or '-'
            ]
            
            label = '_'.join(parts)
            
            # Try to match against events.txt if available
            if events_classes and label in events_classes:
                return label
            elif events_classes:
                # If exact match not found, return the constructed label anyway
                # (it might be a valid combination not in training set)
                return label
            else:
                return label
        
        # Generate annotations for each video
        annotations = {}
        for video in sorted(pred_dict.keys()):
            if use_f3ed:
                coarse_scores, fine_scores, support = pred_dict[video]
                coarse_scores = coarse_scores / support[:, None]
                fine_scores = fine_scores / support[:, None]
                
                # Apply non-maximum suppression
                from util.eval import non_maximum_suppression_np
                coarse_scores = non_maximum_suppression_np(coarse_scores, 5)
                coarse_pred = np.argmax(coarse_scores, axis=1)
                
                # Process fine-grained predictions (same as evaluation code)
                fine_pred = np.zeros_like(fine_scores, int)
                for i in range(len(fine_scores)):
                    # Select max from each element group
                    for start, end in [[0, 2], [2, 5], [5, 8], [16, 24], [25, 29]]:
                        max_idx = np.argmax(fine_scores[i, start:end])
                        fine_pred[i, start + max_idx] = 1
                    if fine_scores[i, 24] > 0.5:  # approach
                        fine_pred[i, 24] = 1
                    if fine_pred[i, 5] != 1:  # not a serve
                        for start, end in [[8, 10], [10, 16]]:
                            max_idx = np.argmax(fine_scores[i, start:end])
                            fine_pred[i, start + max_idx] = 1
                
                # Only keep fine predictions where coarse is foreground
                fine_pred = coarse_pred[:, np.newaxis] * fine_pred
            else:
                scores, support = pred_dict[video]
                scores = scores / support[:, None]  # Normalize by support
                pred = np.argmax(scores, axis=1)
            
            # Extract events
            events = []
            for i in range(len(coarse_pred) if use_f3ed else len(pred)):
                if use_f3ed:
                    if coarse_pred[i] != 0:  # Skip background
                        # Build composite label from fine-grained predictions
                        event_label = build_event_label_from_elements(fine_pred[i], classes_inv)
                        if event_label:
                            # Calculate average score from active elements
                            active_scores = []
                            for idx in range(len(fine_pred[i])):
                                if fine_pred[i, idx] == 1:
                                    active_scores.append(float(fine_scores[i, idx]))
                            score = float(np.mean(active_scores)) if active_scores else float(coarse_scores[i, coarse_pred[i]])
                            
                            events.append({
                                'frame': i,
                                'label': event_label,
                                'score': score
                            })
                else:
                    if pred[i] != 0:  # Skip background
                        score = float(scores[i, pred[i]])
                        events.append({
                            'frame': i,
                            'label': classes_inv[pred[i]],
                            'score': score
                        })
            
            annotations[video] = {
                'events': events,
                'num_frames': len(coarse_pred) if use_f3ed else len(pred),
                'num_events': len(events)
            }
        
        # Update metadata with predictions
        for video_entry in video_metadata:
            video_id = video_entry['video']
            if video_id in annotations:
                video_entry['events'] = annotations[video_id]['events']
        
        # Save annotated metadata
        output_file = os.path.join(output_dir, 'annotations.json')
        with open(output_file, 'w') as f:
            json.dump(video_metadata, f, indent=2)
        
        print(f"\nAnnotations saved to: {output_file}")
        print(f"Total videos annotated: {len(annotations)}")
        
        # Print summary
        total_events = sum(a['num_events'] for a in annotations.values())
        print(f"Total events detected: {total_events}")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_json):
            os.remove(temp_json)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotations using F3-set models'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to video metadata JSON file'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Directory containing extracted frames'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained model (with checkpoint and config.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ncaa_annotations',
        help='Directory to save annotation outputs (default: ./ncaa_annotations)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='f3set-tennis',
        help='Dataset name for loading classes (default: f3set-tennis)'
    )
    parser.add_argument(
        '--use_f3ed',
        action='store_true',
        help='Use F3ED model instead of baseline'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate annotations
    generate_annotations(
        args.metadata,
        args.frame_dir,
        args.model_dir,
        args.output_dir,
        args.dataset,
        args.use_f3ed
    )


if __name__ == '__main__':
    main()
