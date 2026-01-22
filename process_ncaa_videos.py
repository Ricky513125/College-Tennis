#!/usr/bin/env python3
"""
Main script to process NCAA videos and generate annotations using F3-set models.
This script:
1. Detects scenes and splits videos into rally-level clips using pyscenedetect
2. Filters out non-rally scenes (rest, close-ups, replays)
3. Extracts frames from rally clips
4. Creates JSON metadata files
5. Runs inference using F3-set models
6. Generates annotation files
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

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector, ThresholdDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: pyscenedetect not installed. Install with: pip install scenedetect[opencv]")

# Add F3Set to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'F3Set'))


def detect_scenes(video_path, threshold=30.0, min_scene_len=1.0, max_scene_len=60.0):
    """
    Detect scene changes in video using pyscenedetect.
    
    Args:
        video_path: Path to the input video
        threshold: Threshold for content detection (lower = more sensitive)
        min_scene_len: Minimum scene length in seconds (filters very short scenes)
        max_scene_len: Maximum scene length in seconds (filters very long scenes like static shots)
    
    Returns:
        list: List of (start_time, end_time) tuples in seconds
    """
    if not SCENEDETECT_AVAILABLE:
        raise ImportError("pyscenedetect is required for scene detection. Install with: pip install scenedetect[opencv]")
    
    # Create video manager and scene manager
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    
    # Add content detector (detects scene changes based on content differences)
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    
    # Start detection
    video_manager.set_duration()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # Get scene list
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    
    # Filter scenes by length (rally clips are typically 5-30 seconds)
    # Very short scenes (< 2s) are likely cuts/transitions
    # Very long scenes (> 60s) are likely static shots, replays, or breaks
    filtered_scenes = []
    for (start_time, end_time) in scene_list:
        duration = (end_time - start_time).get_seconds()
        if min_scene_len <= duration <= max_scene_len:
            filtered_scenes.append((start_time.get_seconds(), end_time.get_seconds()))
    
    return filtered_scenes


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


def process_videos(video_path, output_dir, frame_dir, enable_scene_detection=True, 
                   scene_threshold=30.0, min_scene_len=2.0, max_scene_len=60.0):
    """
    Process video(s) - either a single video file or all videos in a directory.
    Optionally performs scene detection to split videos into rally-level clips.
    
    Args:
        video_path: Path to a single video file or directory containing videos
        output_dir: Directory to save outputs
        frame_dir: Directory to save extracted frames
        enable_scene_detection: Whether to perform scene detection and split into clips
        scene_threshold: Threshold for scene detection (lower = more sensitive)
        min_scene_len: Minimum scene length in seconds
        max_scene_len: Maximum scene length in seconds
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
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', 'mp4', 'avi', 'mov', 'mkv']
        file_ext = video_path.suffix.lower()
        file_name_lower = video_path.name.lower()
        # Check if extension matches (with or without dot)
        is_video = (file_ext in video_extensions or 
                   any(file_name_lower.endswith(ext) for ext in video_extensions))
        if is_video:
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
    
    if enable_scene_detection and not SCENEDETECT_AVAILABLE:
        print("Warning: Scene detection requested but pyscenedetect not available.")
        print("Falling back to processing entire videos without scene detection.")
        enable_scene_detection = False
    
    # Process each video
    video_metadata = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Create base video ID from filename
        base_video_id = os.path.splitext(video_path.name)[0]
        # Clean video ID (remove special characters that might cause issues)
        base_video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_video_id)
        
        try:
            if enable_scene_detection:
                # Detect scenes and split into clips
                print(f"\nDetecting scenes in {video_path.name}...")
                scenes = detect_scenes(
                    str(video_path), 
                    threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                    max_scene_len=max_scene_len
                )
                print(f"Found {len(scenes)} potential rally clips")
                
                if len(scenes) == 0:
                    print(f"Warning: No scenes detected in {video_path.name}. Processing entire video.")
                    # Fall back to processing entire video
                    video_id = base_video_id
                    num_frames, fps, width, height = extract_frames_from_video(
                        str(video_path), str(frame_dir), video_id
                    )
                    metadata = create_video_metadata(
                        str(video_path), video_id, num_frames, fps, width, height
                    )
                    video_metadata.append(metadata)
                    print(f"Processed {video_path.name}: {num_frames} frames, {fps:.2f} fps")
                else:
                    # Process each detected scene as a separate clip
                    for clip_idx, (start_time, end_time) in enumerate(scenes):
                        clip_id = f"{base_video_id}_clip{clip_idx:03d}"
                        duration = end_time - start_time
                        print(f"  Processing clip {clip_idx+1}/{len(scenes)}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
                        
                        num_frames, fps, width, height = extract_frames_from_video(
                            str(video_path), str(frame_dir), clip_id,
                            start_time=start_time, end_time=end_time
                        )
                        
                        metadata = create_video_metadata(
                            str(video_path), clip_id, num_frames, fps, width, height
                        )
                        video_metadata.append(metadata)
            else:
                # Process entire video without scene detection
                video_id = base_video_id
                num_frames, fps, width, height = extract_frames_from_video(
                    str(video_path), str(frame_dir), video_id
                )
                
                metadata = create_video_metadata(
                    str(video_path), video_id, num_frames, fps, width, height
                )
                video_metadata.append(metadata)
                
                print(f"Processed {video_path.name}: {num_frames} frames, {fps:.2f} fps")
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
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
    parser.add_argument(
        '--no_scene_detection',
        action='store_true',
        help='Disable scene detection and process entire videos (default: enabled)'
    )
    parser.add_argument(
        '--scene_threshold',
        type=float,
        default=30.0,
        help='Scene detection threshold - lower values detect more scenes (default: 30.0)'
    )
    parser.add_argument(
        '--min_scene_len',
        type=float,
        default=2.0,
        help='Minimum scene length in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--max_scene_len',
        type=float,
        default=60.0,
        help='Maximum scene length in seconds (default: 60.0)'
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
    metadata_file = process_videos(
        video_path, 
        output_dir, 
        frame_dir,
        enable_scene_detection=not args.no_scene_detection,
        scene_threshold=args.scene_threshold,
        min_scene_len=args.min_scene_len,
        max_scene_len=args.max_scene_len
    )
    
    if metadata_file:
        print(f"\nNext step: Run inference using:")
        print(f"  python generate_annotations.py --metadata {metadata_file} --frame_dir {frame_dir} --model_dir <path_to_model>")


if __name__ == '__main__':
    main()
