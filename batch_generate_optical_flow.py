#!/usr/bin/env python3
"""
Batch process multiple videos to generate optical flow using RAFT model.
Uses the frames already extracted by F3-set processing.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Import the flow generation function
sys.path.insert(0, os.path.dirname(__file__))
from generate_optical_flow import load_raft_model, process_frames_from_directory


def get_video_list(metadata_file=None, frame_dir=None):
    """
    Get list of videos/rallies to process.
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
        description='Batch generate optical flow for multiple videos using RAFT'
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
        '--video_list',
        type=str,
        nargs='+',
        default=None,
        help='Specific video IDs to process (default: all videos)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Optical Flow Generator (RAFT)")
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
    
    # Load model once
    print("\nLoading RAFT model (this may take a moment)...")
    try:
        raft_model, device = load_raft_model(
            args.model_path,
            small=args.small,
            alternate_corr=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
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
        
        if not os.path.exists(frame_dir):
            print(f"\nWarning: Frame directory not found: {frame_dir}")
            failed.append({'video_id': video_id, 'error': 'Frame directory not found'})
            continue
        
        try:
            print(f"\nProcessing {video_id}...")
            process_frames_from_directory(
                frame_dir,
                video_id,
                raft_model,
                args.output_dir,
                args.skip_frames,
                args.iters
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
