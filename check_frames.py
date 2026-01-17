#!/usr/bin/env python3
"""
Quick script to check if frame files exist and are properly numbered
"""

import os
import sys
import json

def check_frames(metadata_file, frame_dir):
    """Check if frame files exist for all videos"""
    with open(metadata_file, 'r') as f:
        video_metadata = json.load(f)
    
    print(f"Checking {len(video_metadata)} videos...")
    print("=" * 60)
    
    all_ok = True
    for video_entry in video_metadata:
        video_id = video_entry['video']
        expected_frames = video_entry['num_frames']
        video_frame_dir = os.path.join(frame_dir, video_id)
        
        if not os.path.exists(video_frame_dir):
            print(f"❌ {video_id}: Directory not found!")
            print(f"   Expected: {video_frame_dir}")
            all_ok = False
            continue
        
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(video_frame_dir) if f.endswith('.jpg')])
        actual_frames = len(frame_files)
        
        if actual_frames == 0:
            print(f"❌ {video_id}: No frame files found!")
            all_ok = False
            continue
        
        # Check first and last frame
        first_frame = frame_files[0]
        last_frame = frame_files[-1]
        expected_first = '000000.jpg'
        expected_last = f'{expected_frames-1:06d}.jpg'
        
        if first_frame != expected_first:
            print(f"⚠️  {video_id}: First frame mismatch!")
            print(f"   Expected: {expected_first}, Got: {first_frame}")
            all_ok = False
        
        if last_frame != expected_last:
            print(f"⚠️  {video_id}: Last frame mismatch!")
            print(f"   Expected: {expected_last}, Got: {last_frame}")
            print(f"   Expected {expected_frames} frames, got {actual_frames}")
            all_ok = False
        
        if actual_frames != expected_frames:
            print(f"⚠️  {video_id}: Frame count mismatch!")
            print(f"   Expected: {expected_frames}, Got: {actual_frames}")
            all_ok = False
        
        if all_ok or video_metadata.index(video_entry) < 3:
            print(f"✓  {video_id}: {actual_frames} frames OK")
    
    print("=" * 60)
    if all_ok:
        print("All videos have correct frame files!")
    else:
        print("Some videos have issues. Please check the output above.")
    
    return all_ok

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python check_frames.py <metadata.json> <frame_dir>")
        sys.exit(1)
    
    metadata_file = sys.argv[1]
    frame_dir = sys.argv[2]
    check_frames(metadata_file, frame_dir)
