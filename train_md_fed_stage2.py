#!/usr/bin/env python3
"""
Train MD-FED Stage 2 (Multimodal Distillation) for NCAA rally data.
This script adapts MD-FED Stage 2 training to work with rally data structure.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import shutil




def create_data_symlink(data_dir, dataset_name):
    """
    Create symbolic link in MD-FED/data/ to point to our data directory.
    """
    md_fed_data_dir = Path('MD-FED/data') / dataset_name
    target_dir = Path(data_dir) / dataset_name
    
    # Remove existing link or directory
    if md_fed_data_dir.exists() or md_fed_data_dir.is_symlink():
        if md_fed_data_dir.is_symlink():
            md_fed_data_dir.unlink()
        else:
            print(f"Warning: {md_fed_data_dir} exists and is not a symlink. Skipping symlink creation.")
            return False
    
    # Create symlink
    try:
        md_fed_data_dir.symlink_to(os.path.abspath(target_dir))
        print(f"Created symlink: {md_fed_data_dir} -> {target_dir}")
        return True
    except Exception as e:
        print(f"Error creating symlink: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train MD-FED Stage 2 (Multimodal Distillation) for NCAA rally data'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        default='./manual_annotations.json',
        help='Path to manual annotations JSON file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./md_fed_data',
        help='Directory containing prepared data'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Name of the dataset'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        required=True,
        help='Directory containing RGB frames (e.g., /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally)'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        required=True,
        help='Directory containing optical flow files (e.g., /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally)'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        required=True,
        help='Directory containing skeleton pkl files'
    )
    parser.add_argument(
        '--stage1_model_dir',
        type=str,
        default='./md_fed_outputs/stage1',
        help='Directory containing Stage 1 trained model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./md_fed_outputs/stage2',
        help='Directory to save Stage 2 training outputs'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default='rny002_tsm',
        help='Visual architecture'
    )
    parser.add_argument(
        '--skeleton_arch',
        type=str,
        default='stgcn++',
        help='Skeleton architecture'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--clip_len',
        type=int,
        default=96,
        help='Clip length'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=2,
        help='Frame stride'
    )
    parser.add_argument(
        '--prepare_data_only',
        action='store_true',
        help='Only prepare data, do not train'
    )
    
    args = parser.parse_args()
    
    # Step 1: Prepare data
    print("=" * 60)
    print("Step 1: Preparing data for Stage 2 training")
    print("=" * 60)
    
    from prepare_stage2_data import prepare_stage2_data
    data_dir = prepare_stage2_data(
        args.manual_annotations,
        args.data_dir,
        args.dataset_name
    )
    
    # Step 2: Create symlink in MD-FED/data/
    print("\n" + "=" * 60)
    print("Step 2: Creating symlink in MD-FED/data/")
    print("=" * 60)
    create_data_symlink(args.data_dir, args.dataset_name)
    
    if args.prepare_data_only:
        print("\nData preparation complete. Exiting.")
        return
    
    # Step 3: Patch FrameReader to support .npy flow files
    print("\n" + "=" * 60)
    print("Step 3: Patching FrameReader for .npy flow support")
    print("=" * 60)
    from flow_adapter import patch_frame_reader_for_npy_flow
    patch_frame_reader_for_npy_flow()
    
    # Step 4: Call MD-FED training script
    print("\n" + "=" * 60)
    print("Step 4: Starting Stage 2 training")
    print("=" * 60)
    
    # Build command to call MD-FED training script
    # Note: We'll change to MD-FED directory, so use relative path
    md_fed_train_script = 'train_MD-FED.py'
    
    cmd = [
        sys.executable, md_fed_train_script,
        args.dataset_name,
        '--frame_dir', os.path.abspath(args.frame_dir),
        '--flow_dir', os.path.abspath(args.flow_dir),
        '--pose_dir', os.path.abspath(args.pose_dir),
        '--stage', '2',
        '--visual_arch', args.visual_arch,
        '--skeleton_arch', args.skeleton_arch,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--clip_len', str(args.clip_len),
        '--stride', str(args.stride),
        '-s', os.path.abspath(args.output_dir),
        '--num_samples', '-1',
        '--criterion', 'loss'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    
    # Change to MD-FED directory to run the script
    original_dir = os.getcwd()
    try:
        os.chdir('MD-FED')
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        return
    finally:
        os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
