#!/usr/bin/env python3
"""
Wrapper script for MD-FED training that uses data from current directory.
This script patches the get_datasets function to use custom data paths.
"""

import os
import sys

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

# Import after adding to path
from train_MD_FED import get_args, main, get_datasets
import argparse


def get_datasets_patched(args, data_dir):
    """Patched version of get_datasets that uses custom data directory"""
    from util.dataset import load_classes
    from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
    
    # Use custom data directory
    elements_file = os.path.join(data_dir, args.dataset, 'elements.txt')
    train_json = os.path.join(data_dir, args.dataset, 'train.json')
    val_json = os.path.join(data_dir, args.dataset, 'val.json')
    
    classes = load_classes(elements_file)

    if 'f3set-tennis-sub' in args.dataset:
        epoch_num_frames = 500000 if (args.stage == 2 or args.num_samples == -1) else 50000
    elif 'shuttleset' in args.dataset:
        epoch_num_frames = 200000 if (args.stage == 2 or args.num_samples == -1) else 100000
    else:
        epoch_num_frames = 500000 if (args.stage == 2 or args.num_samples == -1) else 100000

    
    dataset_len = epoch_num_frames // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }

    print('Dataset size:', dataset_len)
    num_train_samples, num_val_samples = -1, -1
    if args.num_samples >= 0:
        num_train_samples = int(args.num_samples * 0.8)
        num_val_samples = args.num_samples - num_train_samples
    train_data = ActionSeqDataset(
        classes, train_json,
        args.frame_dir, args.clip_len, dataset_len, is_eval=False, dilate_len=args.dilate_len, stage=args.stage,
        num_samples=num_train_samples, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
        **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, val_json,
        args.frame_dir, args.clip_len, dataset_len // 4, dilate_len=args.dilate_len, stage=args.stage, 
        num_samples=num_val_samples, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'edit':
        # Only perform edit score evaluation during training if criterion is edit
        val_data_frames = ActionSeqVideoDataset(
            classes, val_json,
            args.frame_dir, args.clip_len, overlap_len=0, num_samples=num_val_samples,
            flow_dir=args.flow_dir, pose_dir=args.pose_dir, **dataset_kwargs)

    return classes, train_data, val_data, None, val_data_frames


def main_wrapper():
    """Main function that patches get_datasets"""
    # Parse arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', type=str, default='md_fed_data',
                       help='Directory containing dataset files (default: md_fed_data)')
    known_args, remaining_args = parser.parse_known_args()
    
    # Get original args
    sys.argv = ['train_MD-FED.py'] + remaining_args
    args = get_args()
    
    # Add data_dir to args
    args.data_dir = known_args.data_dir
    
    # Patch get_datasets
    import train_MD_FED
    original_get_datasets = train_MD_FED.get_datasets
    train_MD_FED.get_datasets = lambda a: get_datasets_patched(a, args.data_dir)
    
    # Change to MD-FED directory for imports
    original_cwd = os.getcwd()
    md_fed_path = os.path.join(original_cwd, 'MD-FED')
    if os.path.exists(md_fed_path):
        os.chdir(md_fed_path)
    
    try:
        # Call main
        main(args)
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main_wrapper()
