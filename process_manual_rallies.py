#!/usr/bin/env python3
"""
Script to process videos with manually specified rally time segments.
This script:
1. Reads rally time segments from a JSON configuration file
2. Extracts frames from each rally segment
3. Creates JSON metadata files for each rally
4. Prepares data for F3-set annotation
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


def extract_frames_from_video(video_path, output_dir, video_id, dim=224, start_time=None, end_time=None):
    """
    Extract frames from a video (or a specific time range) and save them in the required format.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save frames
        video_id: Unique identifier for the video
        dim: Height dimension for resizing (default 224)
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = to end)
    
    Returns:
        tuple: (num_frames, fps, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory
    video_frame_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    # Save fps
    with open(os.path.join(video_frame_dir, 'fps.txt'), 'w') as f:
        f.write(str(fps))
    
    # Set start position if specified
    if start_time is not None:
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    count = 0
    end_frame = None
    if end_time is not None:
        end_frame = int(end_time * fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we've reached the end time
        if end_frame is not None:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= end_frame:
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


def create_video_metadata(video_path, video_id, num_frames, fps, width, height, 
                         far_name="Unknown", far_hand="RH", near_name="Unknown", near_hand="RH",
                         far_set=0, far_game=0, far_point=0, near_set=0, near_game=0, near_point=0):
    """
    Create metadata entry for a video in the format expected by F3-set.
    
    Args:
        video_path: Path to the video file
        video_id: Unique identifier
        num_frames: Number of frames
        fps: Frames per second
        width: Video width
        height: Video height
        far_name: Far-end player's name
        far_hand: Far-end player's handedness (RH/LH)
        near_name: Near-end player's name
        near_hand: Near-end player's handedness (RH/LH)
        far_set, far_game, far_point: Far-end player's scores
        near_set, near_game, near_point: Near-end player's scores
    
    Returns:
        dict: Video metadata entry
    """
    return {
        "fps": float(fps),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "video": video_id,
        "far_name": far_name,
        "far_hand": far_hand,
        "far_set": far_set,
        "far_game": far_game,
        "far_point": far_point,
        "near_name": near_name,
        "near_hand": near_hand,
        "near_set": near_set,
        "near_game": near_game,
        "near_point": near_point,
        "events": []  # Will be filled by model predictions
    }


def load_rally_config(config_file):
    """
    Load rally configuration from JSON file.
    
    Supports two formats:
    
    1. Single video format:
    {
        "video_path": "path/to/video.mp4",
        "rallies": [...]
    }
    
    2. Multiple videos format (array):
    [
        {
            "video_path": "path/to/video1.mp4",
            "rallies": [...]
        },
        {
            "video_path": "path/to/video2.mp4",
            "rallies": [...]
        }
    ]
    
    Args:
        config_file: Path to JSON configuration file
    
    Returns:
        list: List of configuration dictionaries (normalized to array format)
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Normalize to array format
    if isinstance(config, list):
        configs = config
    elif isinstance(config, dict):
        # Single video format - convert to array
        configs = [config]
    else:
        raise ValueError("Configuration must be either a dict or a list of dicts")
    
    # Validate each configuration
    validated_configs = []
    for config_idx, config in enumerate(configs):
        if 'video_path' not in config:
            raise ValueError(f"Configuration {config_idx} must contain 'video_path'")
        if 'rallies' not in config:
            raise ValueError(f"Configuration {config_idx} must contain 'rallies' list")
        if not isinstance(config['rallies'], list):
            raise ValueError(f"Configuration {config_idx}: 'rallies' must be a list")
        
        # Validate each rally
        for i, rally in enumerate(config['rallies']):
            if 'start_time' not in rally or 'end_time' not in rally:
                raise ValueError(f"Configuration {config_idx}, Rally {i} must have 'start_time' and 'end_time'")
            if rally['start_time'] >= rally['end_time']:
                raise ValueError(f"Configuration {config_idx}, Rally {i}: start_time ({rally['start_time']}) must be < end_time ({rally['end_time']})")
            if rally['start_time'] < 0:
                raise ValueError(f"Configuration {config_idx}, Rally {i}: start_time must be >= 0")
        
        validated_configs.append(config)
    
    return validated_configs


def process_manual_rallies(config_file, output_dir, frame_dir, base_video_id=None):
    """
    Process video(s) with manually specified rally time segments.
    Supports both single video and multiple videos configuration.
    
    Args:
        config_file: Path to JSON configuration file with rally segments
        output_dir: Directory to save outputs
        frame_dir: Directory to save extracted frames
        base_video_id: Base identifier for video (if None, derived from video filename)
                       Only used for single video format
    
    Returns:
        Path to metadata JSON file
    """
    # Load configuration
    print(f"Loading rally configuration from {config_file}...")
    configs = load_rally_config(config_file)
    
    print(f"Found {len(configs)} video(s) to process")
    
    # Create output directories
    output_dir = Path(output_dir)
    frame_dir = Path(frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all videos
    all_video_metadata = []
    config_dir = Path(config_file).parent
    
    for video_idx, config in enumerate(configs):
        video_path = Path(config['video_path'])
        if not video_path.exists():
            # Try relative to config file directory
            video_path = config_dir / config['video_path']
            if not video_path.exists():
                print(f"Warning: Video file not found: {config['video_path']}, skipping...")
                continue
        
        video_path = str(video_path.resolve())
        
        # Get base video ID
        if base_video_id is None or len(configs) > 1:
            # Use video filename for base ID
            base_video_id_current = os.path.splitext(os.path.basename(video_path))[0]
            # Clean video ID (remove special characters)
            base_video_id_current = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_video_id_current)
        else:
            base_video_id_current = base_video_id
        
        print(f"\n{'='*60}")
        print(f"Processing video {video_idx+1}/{len(configs)}: {os.path.basename(video_path)}")
        print(f"Found {len(config['rallies'])} rally segments")
        
        # Process each rally
        for rally_idx, rally in enumerate(tqdm(config['rallies'], desc=f"  Video {video_idx+1} rallies")):
            rally_id = rally.get('rally_id', f"{base_video_id_current}_rally{rally_idx+1:03d}")
            start_time = rally['start_time']
            end_time = rally['end_time']
            duration = end_time - start_time
            
            # Create full video_id with video name prefix: video_name/rally_id
            full_rally_id = f"{base_video_id_current}/{rally_id}"
            
            try:
                # Extract frames (will be saved to frame_dir/video_name/rally_id/)
                num_frames, fps, width, height = extract_frames_from_video(
                    video_path, str(frame_dir), full_rally_id,
                    start_time=start_time, end_time=end_time
                )
                
                # Create metadata with optional player information
                # Use full_rally_id for video field to match frame directory structure
                metadata = create_video_metadata(
                    video_path, full_rally_id, num_frames, fps, width, height,
                    far_name=rally.get('far_name', 'Unknown'),
                    far_hand=rally.get('far_hand', 'RH'),
                    near_name=rally.get('near_name', 'Unknown'),
                    near_hand=rally.get('near_hand', 'RH'),
                    far_set=rally.get('far_set', 0),
                    far_game=rally.get('far_game', 0),
                    far_point=rally.get('far_point', 0),
                    near_set=rally.get('near_set', 0),
                    near_game=rally.get('near_game', 0),
                    near_point=rally.get('near_point', 0)
                )
                all_video_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error processing {rally_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save metadata JSON
    if len(configs) == 1 and base_video_id:
        metadata_filename = f'{base_video_id}_metadata.json'
    else:
        metadata_filename = 'rallies_metadata.json'
    
    metadata_file = output_dir / metadata_filename
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_video_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Total rallies processed: {len(all_video_metadata)}")
    
    return metadata_file


def create_example_config(output_file):
    """
    Create an example configuration file.
    
    Args:
        output_file: Path to save example configuration
    """
    example_config = {
        "video_path": "../ncaa_videos/example_video.mp4",
        "rallies": [
            {
                "rally_id": "rally_001",
                "start_time": 120.5,
                "end_time": 145.3,
                "far_name": "Player A",
                "far_hand": "RH",
                "near_name": "Player B",
                "near_hand": "LH",
                "far_set": 1,
                "far_game": 2,
                "far_point": 2,
                "near_set": 1,
                "near_game": 2,
                "near_point": 0
            },
            {
                "rally_id": "rally_002",
                "start_time": 200.0,
                "end_time": 225.5
            }
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2, ensure_ascii=False)
    
    print(f"Created example configuration file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process videos with manually specified rally time segments'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON configuration file with rally segments'
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
    parser.add_argument(
        '--base_video_id',
        type=str,
        default=None,
        help='Base identifier for video (default: derived from video filename)'
    )
    parser.add_argument(
        '--create_example',
        type=str,
        default=None,
        help='Create an example configuration file at the specified path'
    )
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example:
        create_example_config(args.create_example)
        return
    
    # Convert to absolute paths
    config_file = os.path.abspath(args.config)
    output_dir = os.path.abspath(args.output_dir)
    frame_dir = os.path.abspath(args.frame_dir)
    
    print(f"Configuration file: {config_file}")
    print(f"Output directory: {output_dir}")
    print(f"Frame directory: {frame_dir}")
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file does not exist: {config_file}")
        return
    
    # Process rallies
    metadata_file = process_manual_rallies(
        config_file, 
        output_dir, 
        frame_dir,
        base_video_id=args.base_video_id
    )
    
    if metadata_file:
        print(f"\nNext step: Run inference using:")
        print(f"  python generate_annotations.py --metadata {metadata_file} --frame_dir {frame_dir} --model_dir <path_to_model>")


if __name__ == '__main__':
    main()
