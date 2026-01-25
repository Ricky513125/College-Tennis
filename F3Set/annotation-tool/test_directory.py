#!/usr/bin/env python3
"""
Test script to check video directory structure and detect issues.
"""

import os
from utils.handle_directory import get_video_directories, get_video_files

def test_directory_structure():
    print("=" * 60)
    print("Testing Video Directory Structure")
    print("=" * 60)
    
    data_path = os.path.join("data", "videos")
    print(f"\nChecking: {os.path.abspath(data_path)}")
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Directory does not exist: {data_path}")
        return
    
    print(f"✓ Directory exists")
    
    # List all items
    try:
        all_items = os.listdir(data_path)
        print(f"\nAll items in {data_path}:")
        for item in sorted(all_items):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/ (directory)")
            elif os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    print(f"  🎬 {item} (video file)")
                else:
                    print(f"  📄 {item} (other file)")
    except PermissionError:
        print(f"❌ ERROR: Permission denied accessing {data_path}")
        return
    
    # Test get_video_directories
    print(f"\n" + "=" * 60)
    print("Testing get_video_directories()")
    print("=" * 60)
    directories = get_video_directories("data")
    print(f"Found {len(directories)} directory option(s):")
    for i, d in enumerate(directories, 1):
        if d == ".":
            print(f"  {i}. '.' (Root - videos directly in data/videos/)")
        else:
            print(f"  {i}. '{d}' (subdirectory)")
    
    # Test get_video_files for each directory
    print(f"\n" + "=" * 60)
    print("Testing get_video_files() for each directory")
    print("=" * 60)
    for directory in directories:
        files = get_video_files("data", directory)
        if directory == ".":
            print(f"\nDirectory: '.' (Root)")
        else:
            print(f"\nDirectory: '{directory}'")
        print(f"  Found {len(files)} video file(s):")
        for f in files[:10]:  # Show first 10
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
    
    print(f"\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    test_directory_structure()
