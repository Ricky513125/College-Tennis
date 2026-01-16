#!/usr/bin/env python3
"""
Complete pipeline to process NCAA videos and generate annotations.
This script orchestrates the entire process:
1. Extract frames from videos
2. Generate annotations using F3-set models
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline for processing NCAA videos and generating annotations'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default='../ncaa_videos',
        help='Directory containing input videos (default: ../ncaa_videos)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained F3-set model (with checkpoint and config.json)'
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
    parser.add_argument(
        '--skip_extraction',
        action='store_true',
        help='Skip frame extraction (use existing frames)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    video_dir = os.path.abspath(args.video_dir)
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)
    frame_dir = os.path.abspath(args.frame_dir)
    
    print("=" * 60)
    print("NCAA Video Annotation Pipeline")
    print("=" * 60)
    print(f"Video directory: {video_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame directory: {frame_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Model type: {'F3ED' if args.use_f3ed else 'Baseline'}")
    print("=" * 60)
    
    # Step 1: Extract frames and create metadata
    if not args.skip_extraction:
        print("\n[Step 1/2] Extracting frames and creating metadata...")
        metadata_file = os.path.join(output_dir, 'ncaa_videos_metadata.json')
        
        cmd = [
            sys.executable,
            'process_ncaa_videos.py',
            '--video_dir', video_dir,
            '--output_dir', output_dir,
            '--frame_dir', frame_dir
        ]
        
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print("Error in frame extraction step")
            return
        
        metadata_file = os.path.join(output_dir, 'ncaa_videos_metadata.json')
        if not os.path.exists(metadata_file):
            print(f"Error: Metadata file not created: {metadata_file}")
            return
    else:
        print("\n[Step 1/2] Skipping frame extraction (using existing frames)...")
        metadata_file = os.path.join(output_dir, 'ncaa_videos_metadata.json')
        if not os.path.exists(metadata_file):
            print(f"Error: Metadata file not found: {metadata_file}")
            print("Please run without --skip_extraction first")
            return
    
    # Step 2: Generate annotations
    print("\n[Step 2/2] Generating annotations...")
    cmd = [
        sys.executable,
        'generate_annotations.py',
        '--metadata', metadata_file,
        '--frame_dir', frame_dir,
        '--model_dir', model_dir,
        '--output_dir', output_dir,
        '--dataset', args.dataset
    ]
    
    if args.use_f3ed:
        cmd.append('--use_f3ed')
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print("Error in annotation generation step")
        return
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"Annotations saved to: {os.path.join(output_dir, 'annotations.json')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
