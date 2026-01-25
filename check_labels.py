#!/usr/bin/env python3
"""
Check the label file used for evaluation to understand why F1 is 0.
"""

import os
import sys
import json
import numpy as np

def check_label_file(label_file):
    """Check label file content and format"""
    
    print("="*60)
    print(f"Checking Label File: {label_file}")
    print("="*60)
    
    if not os.path.exists(label_file):
        print(f"✗ File not found: {label_file}")
        return
    
    print(f"✓ File exists")
    
    # Load and check file
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nTotal videos: {len(data)}")
    
    # Check first few videos
    print("\nFirst 3 videos:")
    for i, video_data in enumerate(data[:3]):
        video_name = video_data.get('video', 'unknown')
        num_frames = video_data.get('num_frames', 0)
        events = video_data.get('events', [])
        
        print(f"\n  Video {i+1}: {video_name}")
        print(f"    Frames: {num_frames}")
        print(f"    Events: {len(events)}")
        
        if events:
            print(f"    First 3 events:")
            for j, event in enumerate(events[:3]):
                frame = event.get('frame', -1)
                label = event.get('label', 'unknown')
                print(f"      Frame {frame}: {label}")
        else:
            print(f"    ⚠ No events in this video!")
    
    # Statistics
    total_events = sum(len(v.get('events', [])) for v in data)
    videos_with_events = sum(1 for v in data if len(v.get('events', [])) > 0)
    videos_without_events = len(data) - videos_with_events
    
    print(f"\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"  Total videos: {len(data)}")
    print(f"  Videos with events: {videos_with_events}")
    print(f"  Videos without events: {videos_without_events}")
    print(f"  Total events: {total_events}")
    print(f"  Average events per video: {total_events / len(data):.2f}")
    
    # Check event frame distribution
    all_event_frames = []
    for video_data in data:
        for event in video_data.get('events', []):
            all_event_frames.append(event.get('frame', -1))
    
    if all_event_frames:
        print(f"\n  Event frame range: {min(all_event_frames)} to {max(all_event_frames)}")
        print(f"  Event frame distribution:")
        print(f"    Min: {min(all_event_frames)}")
        print(f"    Max: {max(all_event_frames)}")
        print(f"    Mean: {np.mean(all_event_frames):.1f}")
        print(f"    Median: {np.median(all_event_frames):.1f}")
    
    # Check event labels
    event_labels = {}
    for video_data in data:
        for event in video_data.get('events', []):
            label = event.get('label', 'unknown')
            event_labels[label] = event_labels.get(label, 0) + 1
    
    print(f"\n  Event label distribution (top 10):")
    sorted_labels = sorted(event_labels.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels[:10]:
        print(f"    {label}: {count}")
    
    # Check if labels match expected format
    print(f"\n" + "="*60)
    print("Label Format Check:")
    print("="*60)
    
    # Check required fields
    required_fields = ['video', 'num_frames', 'fps', 'events']
    missing_fields = []
    for i, video_data in enumerate(data[:10]):  # Check first 10
        for field in required_fields:
            if field not in video_data:
                missing_fields.append(f"Video {i}: missing '{field}'")
    
    if missing_fields:
        print("  ⚠ Missing required fields:")
        for msg in missing_fields[:5]:
            print(f"    {msg}")
    else:
        print("  ✓ All required fields present")
    
    # Check event format
    event_format_ok = True
    for i, video_data in enumerate(data[:10]):
        for event in video_data.get('events', []):
            if 'frame' not in event or 'label' not in event:
                print(f"  ⚠ Video {i}: event missing 'frame' or 'label'")
                event_format_ok = False
                break
    
    if event_format_ok:
        print("  ✓ Event format is correct")
    
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)
    
    if videos_without_events > len(data) * 0.5:
        print("\n⚠ More than 50% of videos have no events")
        print("  This could cause F1=0 if model predicts events in these videos")
    
    if total_events == 0:
        print("\n🔴 CRITICAL: No events in label file!")
        print("  This would definitely cause F1=0")
    
    if total_events > 0 and videos_with_events < len(data) * 0.1:
        print("\n⚠ Very few videos have events")
        print("  Model may be predicting events in videos without labels")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # Default path based on run_md_fed_stage1.py
    default_label_file = 'md_fed_data/f3set-tennis-sub/val.json'
    
    if len(sys.argv) > 1:
        label_file = sys.argv[1]
    else:
        label_file = default_label_file
    
    check_label_file(label_file)
