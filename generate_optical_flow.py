#!/usr/bin/env python3
"""
Generate optical flow files for videos using RAFT model.
Uses the frames already extracted by F3-set processing.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add RAFT to path
raft_path = os.path.join(os.path.dirname(__file__), 'RAFT')
sys.path.insert(0, os.path.join(raft_path, 'core'))

from raft import RAFT
from utils.utils import InputPadder


def load_image(imfile, device='cuda'):
    """Load image and convert to tensor"""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def save_flow(flow, output_path):
    """Save optical flow to file"""
    # Flow shape: [2, H, W]
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    
    # Save as .npy file (numpy format)
    np.save(output_path, flow_np)
    
    return flow_np


def process_frames_from_directory(frame_dir, video_id, raft_model, output_dir, 
                                   skip_frames=1, iters=20):
    """
    Process frames from a directory and generate optical flow.
    
    Args:
        frame_dir: Directory containing frame images
        video_id: Video identifier
        raft_model: Loaded RAFT model
        output_dir: Directory to save flow files
        skip_frames: Process every N frames (1 = all frames)
        iters: Number of RAFT iterations
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if len(frame_files) < 2:
        raise ValueError(f"Need at least 2 frames, found {len(frame_files)}")
    
    # Read FPS if available
    fps_file = os.path.join(frame_dir, 'fps.txt')
    if os.path.exists(fps_file):
        with open(fps_file, 'r') as f:
            fps = float(f.read().strip())
    else:
        fps = 30.0
        print(f"Warning: fps.txt not found, using default {fps} fps")
    
    # Get image dimensions from first frame
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        raise ValueError(f"Could not read first frame: {first_frame_path}")
    height, width = first_img.shape[:2]
    
    total_frames = len(frame_files)
    print(f"Frame directory: {frame_dir}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Size: {width}x{height}")
    print(f"Processing every {skip_frames} frame(s)")
    
    # Create output directory for this video
    video_flow_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_flow_dir, exist_ok=True)
    
    # Process frame pairs
    flow_files = []
    processed_count = 0
    
    # Filter frames based on skip_frames
    filtered_frames = [frame_files[i] for i in range(0, len(frame_files), skip_frames)]
    
    pbar = tqdm(total=len(filtered_frames)-1, desc=f"Processing {video_id}")
    
    raft_model.eval()
    with torch.no_grad():
        for i in range(len(filtered_frames) - 1):
            frame1_file = filtered_frames[i]
            frame2_file = filtered_frames[i + 1]
            
            frame1_path = os.path.join(frame_dir, frame1_file)
            frame2_path = os.path.join(frame_dir, frame2_file)
            
            # Load images
            image1 = load_image(frame1_path, device)
            image2 = load_image(frame2_path, device)
            
            # Pad images to be divisible by 8
            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)
            
            # Compute optical flow
            flow_low, flow_up = raft_model(image1_padded, image2_padded, iters=iters, test_mode=True)
            
            # Unpad flow
            flow_up = padder.unpad(flow_up)
            
            # Get frame numbers
            frame1_num = int(frame1_file.replace('.jpg', ''))
            frame2_num = int(frame2_file.replace('.jpg', ''))
            
            # Save flow file (flow from frame1 to frame2)
            flow_filename = f'{frame1_num:06d}_{frame2_num:06d}.npy'
            flow_path = os.path.join(video_flow_dir, flow_filename)
            flow_np = save_flow(flow_up, flow_path)
            
            flow_files.append({
                'frame1': frame1_num,
                'frame2': frame2_num,
                'flow_file': flow_filename,
                'flow_shape': list(flow_np.shape)
            })
            
            processed_count += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save metadata
    metadata = {
        'video_id': video_id,
        'frame_dir': str(frame_dir),
        'fps': float(fps),
        'total_frames': total_frames,
        'processed_pairs': processed_count,
        'width': width,
        'height': height,
        'flow_files': flow_files
    }
    
    metadata_path = os.path.join(video_flow_dir, 'flow_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nOptical flow saved to: {video_flow_dir}")
    print(f"Processed {processed_count} frame pairs")
    print(f"Metadata saved to: {metadata_path}")
    
    return video_flow_dir


def load_raft_model(model_path, small=True, alternate_corr=False):
    """Load RAFT model"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create args object that supports 'in' operator (as used by RAFT)
    # RAFT checks 'dropout' not in args, so we need a dict-like or custom class
    class Args:
        def __init__(self):
            self.small = small
            self.alternate_corr = alternate_corr
            self.mixed_precision = False
            self.dropout = 0
        
        def __contains__(self, key):
            """Support 'in' operator for checking attributes"""
            return hasattr(self, key)
    
    args = Args()
    
    print(f"Loading RAFT model from: {model_path}")
    print(f"Model type: {'small' if small else 'standard'}")
    
    # Load model
    model = RAFT(args)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # RAFT checkpoints are typically saved as state_dict directly
    # or wrapped in DataParallel
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Direct state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper (RAFT demo uses DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load with strict=False to handle any minor mismatches
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, device


def main():
    parser = argparse.ArgumentParser(
        description='Generate optical flow for videos using RAFT model'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to frame directory OR video file'
    )
    parser.add_argument(
        '--video_id',
        type=str,
        default=None,
        help='Video ID (required when using frame directory)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default=None,
        help='Base frame directory (default: ./ncaa_frames)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./RAFT/models/raft-small.pth',
        help='Path to RAFT model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ncaa_optical_flow',
        help='Directory to save optical flow files'
    )
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=1,
        help='Process every N frames (default: 1 = all frames)'
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=20,
        help='Number of RAFT iterations (default: 20)'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        default=True,
        help='Use small RAFT model (default: True)'
    )
    parser.add_argument(
        '--alternate_corr',
        action='store_true',
        help='Use alternate correlation implementation'
    )
    parser.add_argument(
        '--use_frames',
        action='store_true',
        help='Force using frame directory mode'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    if args.use_frames or os.path.isdir(args.input_path):
        # Frame directory mode
        if args.frame_dir:
            frame_dir = os.path.join(args.frame_dir, args.video_id or os.path.basename(args.input_path))
        else:
            frame_dir = args.input_path
        
        if not os.path.exists(frame_dir):
            print(f"Error: Frame directory not found: {frame_dir}")
            return
        
        video_id = args.video_id or os.path.basename(frame_dir)
        input_type = 'frames'
    else:
        print("Error: Video file mode not yet implemented. Please use frame directory mode.")
        print("Use --use_frames flag with frame directory path.")
        return
    
    if args.output_dir is None:
        args.output_dir = './ncaa_optical_flow'
    
    print("=" * 60)
    print("RAFT Optical Flow Generator")
    print("=" * 60)
    print(f"Frame directory: {frame_dir}")
    print(f"Video ID: {video_id}")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip frames: {args.skip_frames}")
    print("=" * 60)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Load model
    print("\nLoading RAFT model...")
    try:
        raft_model, device = load_raft_model(
            args.model_path,
            small=args.small,
            alternate_corr=args.alternate_corr
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process frames
    print("\nProcessing frames...")
    try:
        process_frames_from_directory(
            frame_dir,
            video_id,
            raft_model,
            args.output_dir,
            args.skip_frames,
            args.iters
        )
    except Exception as e:
        print(f"Error processing frames: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Optical flow generation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
