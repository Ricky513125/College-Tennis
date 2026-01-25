#!/usr/bin/env python3
"""
Prepare data for MD-FED Stage 2 training from manual annotations.
This script creates the necessary JSON files and data structure for Stage 2 training.
"""

import os
import json
import argparse
from pathlib import Path
import shutil

def prepare_stage2_data(manual_annotations_file, output_dir, dataset_name='ncaa-rally'):
    """
    Prepare data for Stage 2 training from manual annotations.
    
    Args:
        manual_annotations_file: Path to manual_annotations.json
        output_dir: Directory to save prepared data
        dataset_name: Name of the dataset
    """
    # Load manual annotations
    print(f"Loading manual annotations from {manual_annotations_file}...")
    with open(manual_annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} rally annotations")
    
    # Create output directory structure
    data_dir = Path(output_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # For Stage 2, we don't need labels, but we need the video metadata
    # Create train.json and val.json with all rallies (Stage 2 is unsupervised)
    # We can use all data for training, and split some for validation
    
    # Split: 80% train, 20% val
    split_idx = int(len(annotations) * 0.8)
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    # Save train.json
    train_file = data_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_annotations)} rallies to {train_file}")
    
    # Save val.json
    val_file = data_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_annotations)} rallies to {val_file}")
    
    # Copy elements.txt and events.txt from MD-FED if they exist
    md_fed_data_dir = Path('MD-FED/data/f3set-tennis-sub')
    if md_fed_data_dir.exists():
        for file_name in ['elements.txt', 'events.txt']:
            src_file = md_fed_data_dir / file_name
            if src_file.exists():
                dst_file = data_dir / file_name
                shutil.copy2(src_file, dst_file)
                print(f"Copied {file_name} to {dst_file}")
            else:
                print(f"Warning: {file_name} not found in {md_fed_data_dir}")
    else:
        print(f"Warning: MD-FED data directory not found: {md_fed_data_dir}")
        print("You may need to manually copy elements.txt and events.txt")
    
    print(f"\nData preparation complete!")
    print(f"Data directory: {data_dir}")
    print(f"Train rallies: {len(train_annotations)}")
    print(f"Val rallies: {len(val_annotations)}")
    
    return data_dir


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for MD-FED Stage 2 training'
    )
    parser.add_argument(
        '--manual_annotations',
        type=str,
        default='./manual_annotations.json',
        help='Path to manual annotations JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./md_fed_data',
        help='Directory to save prepared data'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncaa-rally',
        help='Name of the dataset'
    )
    
    args = parser.parse_args()
    
    prepare_stage2_data(
        args.manual_annotations,
        args.output_dir,
        args.dataset_name
    )


if __name__ == '__main__':
    main()
