#!/usr/bin/env python3
"""
Prepare data for MD-FED training by merging train.json and val.json,
and setting up the correct directory structure.
"""

import os
import json
import argparse
import shutil
from pathlib import Path


def merge_json_files(json_files, output_file):
    """Merge multiple JSON files into one"""
    merged_data = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping...")
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)
        print(f"Loaded {len(data) if isinstance(data, list) else 1} entries from {json_file}")
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged {len(merged_data)} entries into {output_file}")
    return merged_data


def setup_md_fed_data(
    tennis_dir,
    output_dir,
    dataset_name='f3set-tennis-sub',
    use_test_as_val=True
):
    """
    Setup MD-FED data directory structure
    
    Args:
        tennis_dir: Path to ~/Tennis directory containing train.json, val.json, test.json
        output_dir: Path to output directory (will create subdirectory for dataset)
        dataset_name: Name of the dataset subdirectory
        use_test_as_val: If True, use test.json as validation set
    """
    # Create dataset directory in output_dir
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Paths to source JSON files
    train_json = os.path.join(tennis_dir, 'train.json')
    val_json = os.path.join(tennis_dir, 'val.json')
    test_json = os.path.join(tennis_dir, 'test.json')
    
    # Merge train.json + val.json for training
    merged_train_json = os.path.join(dataset_dir, 'train.json')
    if os.path.exists(train_json) and os.path.exists(val_json):
        print("Merging train.json and val.json for training data...")
        merge_json_files([train_json, val_json], merged_train_json)
    elif os.path.exists(train_json):
        print("Copying train.json (val.json not found)...")
        shutil.copy(train_json, merged_train_json)
    else:
        raise FileNotFoundError(f"train.json not found in {tennis_dir}")
    
    # Use test.json as validation set
    val_output_json = os.path.join(dataset_dir, 'val.json')
    if use_test_as_val:
        if os.path.exists(test_json):
            print(f"Using test.json as validation set...")
            shutil.copy(test_json, val_output_json)
        else:
            print(f"Warning: test.json not found, using original val.json if available...")
            if os.path.exists(val_json):
                shutil.copy(val_json, val_output_json)
            else:
                raise FileNotFoundError(f"Neither test.json nor val.json found in {tennis_dir}")
    else:
        if os.path.exists(val_json):
            shutil.copy(val_json, val_output_json)
        else:
            raise FileNotFoundError(f"val.json not found in {tennis_dir}")
    
    # Copy test.json if it exists (for final evaluation)
    test_output_json = os.path.join(dataset_dir, 'test.json')
    if os.path.exists(test_json):
        shutil.copy(test_json, test_output_json)
        print(f"Copied test.json to {test_output_json}")
    
    # Copy elements.txt and events.txt from MD-FED/data if they exist
    md_fed_elements = os.path.join('MD-FED', 'data', dataset_name, 'elements.txt')
    md_fed_events = os.path.join('MD-FED', 'data', dataset_name, 'events.txt')
    
    elements_file = os.path.join(dataset_dir, 'elements.txt')
    events_file = os.path.join(dataset_dir, 'events.txt')
    
    if os.path.exists(md_fed_elements) and not os.path.exists(elements_file):
        print(f"Copying elements.txt from {md_fed_elements}...")
        shutil.copy(md_fed_elements, elements_file)
    elif not os.path.exists(elements_file):
        print(f"Warning: elements.txt not found. Please copy it to {elements_file}")
    
    if os.path.exists(md_fed_events) and not os.path.exists(events_file):
        print(f"Copying events.txt from {md_fed_events}...")
        shutil.copy(md_fed_events, events_file)
    elif not os.path.exists(events_file):
        print(f"Warning: events.txt not found. Please copy it to {events_file}")
    
    print(f"\nData setup complete!")
    print(f"Training data: {merged_train_json}")
    print(f"Validation data: {val_output_json}")
    print(f"Dataset directory: {dataset_dir}")
    
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for MD-FED training'
    )
    parser.add_argument(
        '--tennis_dir',
        type=str,
        required=True,
        help='Path to ~/Tennis directory containing train.json, val.json, test.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='md_fed_data',
        help='Output directory for prepared data (default: md_fed_data, will create subdirectory for dataset)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='f3set-tennis-sub',
        help='Dataset name (default: f3set-tennis-sub)'
    )
    parser.add_argument(
        '--use_test_as_val',
        action='store_true',
        default=True,
        help='Use test.json as validation set (default: True)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    tennis_dir = os.path.abspath(args.tennis_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(tennis_dir):
        raise FileNotFoundError(f"Tennis directory not found: {tennis_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    setup_md_fed_data(
        tennis_dir,
        output_dir,
        args.dataset,
        args.use_test_as_val
    )


if __name__ == '__main__':
    main()
