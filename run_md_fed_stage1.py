#!/usr/bin/env python3
"""
Helper script to run MD-FED Stage 1 training with data in current directory.
This script sets up symbolic links and runs the training.
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path


def setup_data_links(data_dir, dataset_name='f3set-tennis-sub'):
    """Create symbolic links in MD-FED/data to point to current directory data"""
    md_fed_data_dir = os.path.join('MD-FED', 'data', dataset_name)
    
    # Create MD-FED/data directory if it doesn't exist
    os.makedirs(os.path.dirname(md_fed_data_dir), exist_ok=True)
    
    # Remove existing link or directory if it exists
    if os.path.exists(md_fed_data_dir):
        if os.path.islink(md_fed_data_dir):
            os.unlink(md_fed_data_dir)
        elif os.path.isdir(md_fed_data_dir):
            print(f"Warning: {md_fed_data_dir} already exists as a directory")
            print("Please remove it manually or use a different dataset name")
            return False
    
    # Create symbolic link
    source_dir = os.path.abspath(os.path.join(data_dir, dataset_name))
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        print("Please run prepare_md_fed_data.py first")
        return False
    
    try:
        os.symlink(source_dir, md_fed_data_dir)
        print(f"Created symbolic link: {md_fed_data_dir} -> {source_dir}")
        return True
    except OSError as e:
        print(f"Error creating symbolic link: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run MD-FED Stage 1 training with data in current directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='md_fed_data',
        help='Directory containing prepared data (default: md_fed_data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='f3set-tennis-sub',
        help='Dataset name (default: f3set-tennis-sub)'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        required=True,
        help='Path to skeleton pkl files directory'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default='frames',
        help='Path to RGB frames directory (default: frames)'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        default='flows',
        help='Path to optical flow directory (default: flows)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='md_fed_outputs/stage1',
        help='Output directory for checkpoints and logs (default: md_fed_outputs/stage1)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default='rny002_tsm',
        help='Visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--skeleton_arch',
        type=str,
        default='stgcn++',
        help='Skeleton architecture (default: stgcn++)'
    )
    parser.add_argument(
        '--setup_only',
        action='store_true',
        help='Only setup data links, do not run training'
    )
    
    args = parser.parse_args()
    
    # Setup symbolic links
    print("Setting up data links...")
    if not setup_data_links(args.data_dir, args.dataset):
        sys.exit(1)
    
    if args.setup_only:
        print("Data links setup complete. Run training manually.")
        return
    
    # Prepare training command
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        'train_MD-FED.py',
        args.dataset,
        '--frame_dir', args.frame_dir,
        '--flow_dir', args.flow_dir,
        '--pose_dir', os.path.abspath(args.pose_dir),
        '--stage', '1',
        '--visual_arch', args.visual_arch,
        '--skeleton_arch', args.skeleton_arch,
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '-s', output_dir
    ]
    
    print("\n" + "="*60)
    print("Starting MD-FED Stage 1 training...")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.path.join(os.getcwd(), 'MD-FED')}")
    print("="*60 + "\n")
    
    # Change to MD-FED directory and run
    md_fed_dir = os.path.join(os.getcwd(), 'MD-FED')
    if not os.path.exists(md_fed_dir):
        print(f"Error: MD-FED directory not found: {md_fed_dir}")
        sys.exit(1)
    
    try:
        subprocess.run(cmd, cwd=md_fed_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
