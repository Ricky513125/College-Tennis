#!/usr/bin/env python3
"""
Extract frames from a single video file.
Useful for re-processing videos that were interrupted or have missing frames.
"""

import os
import sys
import json
import argparse
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    
    # Create output directory
    video_frame_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_frame_dir, exist_ok=True)
    
    # Save fps
    with open(os.path.join(video_frame_dir, 'fps.txt'), 'w') as f:
        f.write(str(fps))
    
    # Check existing frames
    existing_frames = set()
    if os.path.exists(video_frame_dir):
        for f in os.listdir(video_frame_dir):
            if f.endswith('.jpg'):
                try:
                    frame_num = int(f.replace('.jpg', ''))
                    existing_frames.add(frame_num)
                except:
                    pass
    
    print(f"Found {len(existing_frames)} existing frames")
    
    # Extract frames
    count = 0
    extracted = 0
    skipped = 0
    
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc=f"Extracting {video_id}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if frame already exists
        if count in existing_frames:
            skipped += 1
            count += 1
            pbar.update(1)
            continue
        
        # Resize frame
        H, W, _ = frame.shape
        resized = cv2.resize(frame, (W * dim // H, dim))
        
        # Save frame
        frame_path = os.path.join(video_frame_dir, f'{count:06d}.jpg')
        cv2.imwrite(frame_path, resized)
        extracted += 1
        count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    print(f"\nExtraction complete:")
    print(f"  Total frames: {count}")
    print(f"  Newly extracted: {extracted}")
    print(f"  Skipped (already existed): {skipped}")
    
    return count, fps, width, height


def update_metadata(metadata_file, video_id, num_frames, fps, width, height):
    """Update metadata for a single video"""
    with open(metadata_file, 'r') as f:
        video_metadata = json.load(f)
    
    # Find and update the video entry
    updated = False
    for entry in video_metadata:
        if entry['video'] == video_id:
            entry['num_frames'] = num_frames
            entry['fps'] = float(fps)
            entry['width'] = width
            entry['height'] = height
            updated = True
            print(f"\nUpdated metadata for {video_id}")
            break
    
    if not updated:
        print(f"\nWarning: Video {video_id} not found in metadata")
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(video_metadata, f, indent=2)
    
    return updated


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from a single video file'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file'
    )
    parser.add_argument(
        '--video_id',
        type=str,
        default=None,
        help='Video ID (default: filename without extension)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default='./ncaa_frames',
        help='Directory to save extracted frames (default: ./ncaa_frames)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to metadata JSON file to update (optional)'
    )
    parser.add_argument(
        '--dim',
        type=int,
        default=224,
        help='Frame height dimension (default: 224)'
    )
    
    args = parser.parse_args()
    
    # Get video ID
    if args.video_id is None:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
        # Clean video ID (remove special characters)
        video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
    else:
        video_id = args.video_id
    
    print(f"Video ID: {video_id}")
    print(f"Video path: {args.video_path}")
    print(f"Frame directory: {args.frame_dir}")
    print("=" * 60)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Extract frames
    try:
        num_frames, fps, width, height = extract_frames_from_video(
            args.video_path,
            args.frame_dir,
            video_id,
            args.dim
        )
        
        # Update metadata if provided
        if args.metadata and os.path.exists(args.metadata):
            update_metadata(args.metadata, video_id, num_frames, fps, width, height)
        
        print("\n" + "=" * 60)
        print("Frame extraction completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
