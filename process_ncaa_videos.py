#!/usr/bin/env python3
"""
Main script to process NCAA videos and generate annotations using F3-set models.
This script:
1. Extracts frames from videos in ncaa_videos folder
2. Creates JSON metadata files
3. Runs inference using F3-set models
4. Generates annotation files
"""

import os
import sys
import json
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add F3Set to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F3Set'))

def extract_frames_from_video(video_path, output_dir, video_id, dim=224):
    """
    Extract frames from a video and save them in the required format.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save frames
        video_id: Unique identifier for the video
        dim: Height dimension for resizing (default 224)
    
    Returns:
        tuple: (num_frames, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    video_frame_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    # Save fps
    with open(os.path.join(video_frame_dir, 'fps.txt'), 'w') as f:
        f.write(str(fps))
    
    # Extract frames
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        H, W, _ = frame.shape
        resized = cv2.resize(frame, (W * dim // H, dim))
        
        # Save frame
        frame_path = os.path.join(video_frame_dir, f'{count:06d}.jpg')
        cv2.imwrite(frame_path, resized)
        count += 1
    
    cap.release()
    return count, fps, width, height


def create_video_metadata(video_path, video_id, num_frames, fps, width, height):
    """
    Create metadata entry for a video in the format expected by F3-set.
    
    Args:
        video_path: Path to the video file
        video_id: Unique identifier
        num_frames: Number of frames
        fps: Frames per second
        width: Video width
        height: Video height
    
    Returns:
        dict: Video metadata entry
    """
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    return {
        "fps": float(fps),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "video": video_id,
        "far_name": "Unknown",
        "far_hand": "RH",
        "far_set": 0,
        "far_game": 0,
        "far_point": 0,
        "near_name": "Unknown",
        "near_hand": "RH",
        "near_set": 0,
        "near_game": 0,
        "near_point": 0,
        "events": []  # Will be filled by model predictions
    }


def process_videos(video_path, output_dir, frame_dir):
    """
    Process video(s) - either a single video file or all videos in a directory.
    
    Args:
        video_path: Path to a single video file or directory containing videos
        output_dir: Directory to save outputs
        frame_dir: Directory to save extracted frames
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frame_dir = Path(frame_dir)
    
    # Create output directories
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if it's a file or directory
    if video_path.is_file():
        # Single video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if video_path.suffix.lower() in video_extensions:
            video_files = [video_path]
        else:
            print(f"Error: {video_path} is not a supported video file")
            return None
    elif video_path.is_dir():
        # Directory - find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(video_path.glob(f'*{ext}')))
            video_files.extend(list(video_path.glob(f'*{ext.upper()}')))
        
        if not video_files:
            print(f"No video files found in {video_path}")
            return None
    else:
        print(f"Error: {video_path} does not exist")
        return None
    
    print(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    video_metadata = []
    for video_path in tqdm(video_files, desc="Extracting frames"):
        # Create video ID from filename
        video_id = os.path.splitext(video_path.name)[0]
        # Clean video ID (remove special characters that might cause issues)
        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
        
        try:
            # Extract frames
            num_frames, fps, width, height = extract_frames_from_video(
                str(video_path), str(frame_dir), video_id
            )
            
            # Create metadata
            metadata = create_video_metadata(
                str(video_path), video_id, num_frames, fps, width, height
            )
            video_metadata.append(metadata)
            
            print(f"Processed {video_path.name}: {num_frames} frames, {fps:.2f} fps")
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            continue
    
    # Save metadata JSON
    metadata_file = output_dir / 'ncaa_videos_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(video_metadata, f, indent=2)
    
    print(f"\nSaved metadata to {metadata_file}")
    print(f"Total videos processed: {len(video_metadata)}")
    
    return metadata_file


def main():
    parser = argparse.ArgumentParser(
        description='Process NCAA videos and prepare for F3-set annotation'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default='../ncaa_videos',
        help='Path to a single video file or directory containing input videos (default: ../ncaa_videos)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ncaa_annotations',
        help='Directory to save outputs (default: ./ncaa_annotations)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default='./ncaa_frames',
        help='Directory to save extracted frames (default: ./ncaa_frames)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    video_path = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    frame_dir = os.path.abspath(args.frame_dir)
    
    print(f"Video path: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame directory: {frame_dir}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        return
    
    # Process videos
    metadata_file = process_videos(video_path, output_dir, frame_dir)
    
    if metadata_file:
        print(f"\nNext step: Run inference using:")
        print(f"  python generate_annotations.py --metadata {metadata_file} --frame_dir {frame_dir} --model_dir <path_to_model>")


if __name__ == '__main__':
    main()
