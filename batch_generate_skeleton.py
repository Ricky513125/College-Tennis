#!/usr/bin/env python3
"""
Batch process multiple videos to generate skeleton annotations using extracted frames.
This script uses the frames already extracted by F3-set processing.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Import the skeleton generation function
sys.path.insert(0, os.path.dirname(__file__))
from generate_skeleton_annotations import load_models, process_frames_from_directory


def get_video_list(metadata_file=None, frame_dir=None):
    """
    Get list of videos/rallies to process.
    Can be from metadata file or by scanning frame directory.
    Supports both flat structure (video_id) and nested structure (video_name/rally_id).
    """
    videos = []
    
    if metadata_file and os.path.exists(metadata_file):
        # Get from metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        for entry in metadata:
            video_id = entry['video']
            videos.append({
                'video_id': video_id,
                'num_frames': entry.get('num_frames', 0)
            })
    elif frame_dir and os.path.exists(frame_dir):
        # Scan frame directory (supports nested structure: video_name/rally_id)
        for video_name in os.listdir(frame_dir):
            video_path = os.path.join(frame_dir, video_name)
            if os.path.isdir(video_path):
                # Check if it's a nested structure (video_name/rally_id)
                has_nested = False
                for item in os.listdir(video_path):
                    item_path = os.path.join(video_path, item)
                    if os.path.isdir(item_path):
                        # This is a nested structure: video_name/rally_id
                        frame_files = [f for f in os.listdir(item_path) if f.endswith('.jpg')]
                        if frame_files:
                            has_nested = True
                            rally_id = item
                            full_video_id = f"{video_name}/{rally_id}"
                            videos.append({
                                'video_id': full_video_id,
                                'num_frames': len(frame_files)
                            })
                
                if not has_nested:
                    # Flat structure: just video_id
                    frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                    if frame_files:
                        videos.append({
                            'video_id': video_name,
                            'num_frames': len(frame_files)
                        })
    else:
        raise ValueError("Either metadata_file or frame_dir must be provided")
    
    return videos


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate skeleton annotations for multiple videos using extracted frames'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='./ncaa_annotations/ncaa_videos_metadata.json',
        help='Path to video metadata JSON file'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default='./ncaa_frames',
        help='Base directory containing frame directories'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth',
        help='Path to HRNet model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml',
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ncaa_skeleton_annotations',
        help='Directory to save skeleton annotation JSON files'
    )
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=1,
        help='Process every N frames (default: 1 = all frames)'
    )
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.9,
        help='Person detection threshold (default: 0.9)'
    )
    parser.add_argument(
        '--video_list',
        type=str,
        nargs='+',
        default=None,
        help='Specific video IDs to process (default: all videos)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Skeleton Annotation Generator")
    print("=" * 60)
    print(f"Frame directory: {args.frame_dir}")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip frames: {args.skip_frames}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video list
    print("\nLoading video list...")
    try:
        videos = get_video_list(args.metadata, args.frame_dir)
    except Exception as e:
        print(f"Error loading video list: {e}")
        return
    
    # Filter by video_list if specified
    if args.video_list:
        videos = [v for v in videos if v['video_id'] in args.video_list]
        print(f"Processing {len(videos)} specified videos")
    else:
        print(f"Found {len(videos)} videos to process")
    
    if not videos:
        print("No videos to process!")
        return
    
    # Load models once
    print("\nLoading models (this may take a moment)...")
    try:
        pose_model, box_model, device = load_models(args.model_path, args.config)
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process each video
    print("\nProcessing videos...")
    successful = []
    failed = []
    
    for video in tqdm(videos, desc="Videos"):
        video_id = video['video_id']
        # Handle nested structure (video_name/rally_id)
        frame_dir = os.path.join(args.frame_dir, video_id)
        # Create safe output filename (replace / with _)
        safe_video_id = video_id.replace('/', '_')
        output_file = os.path.join(args.output_dir, f'{safe_video_id}_skeleton.json')
        
        if not os.path.exists(frame_dir):
            print(f"\nWarning: Frame directory not found: {frame_dir}")
            failed.append({'video_id': video_id, 'error': 'Frame directory not found'})
            continue
        
        try:
            print(f"\nProcessing {video_id}...")
            process_frames_from_directory(
                frame_dir,
                video_id,
                pose_model,
                box_model,
                output_file,
                args.detection_threshold,
                args.skip_frames
            )
            successful.append(video_id)
        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            failed.append({'video_id': video_id, 'error': str(e)})
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total videos: {len(videos)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully processed videos:")
        for vid in successful:
            print(f"  ✓ {vid}")
    
    if failed:
        print(f"\nFailed videos:")
        for item in failed:
            print(f"  ✗ {item['video_id']}: {item['error']}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
