#!/usr/bin/env python3
"""
Manually run evaluation on a checkpoint to see what the model is predicting.
This helps diagnose why F1 scores are 0.
"""

import os
import sys
import torch
import numpy as np

def check_model_outputs(checkpoint_path, output_dir):
    """Load a checkpoint and check what it's predicting"""
    
    print("="*60)
    print("Checking Model Predictions")
    print("="*60)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"✓ Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Keys: {list(checkpoint.keys())[:10]}...")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Try to find error_sequences.txt
    error_files = [
        'error_sequences.txt',
        os.path.join(output_dir, 'error_sequences.txt'),
        'MD-FED/error_sequences.txt',
        os.path.join(os.path.dirname(output_dir), 'error_sequences.txt'),
    ]
    
    for error_file in error_files:
        if os.path.exists(error_file):
            print(f"\n✓ Found error_sequences.txt: {error_file}")
            with open(error_file, 'r') as f:
                content = f.read()
            
            if len(content.strip()) == 0:
                print("  ⚠ File is EMPTY - model predicts no events")
            else:
                lines = content.strip().split('\n')
                print(f"  File has {len(lines)} lines")
                print("  First 20 lines:")
                for i, line in enumerate(lines[:20]):
                    if line.strip():
                        print(f"    {i+1}: {line[:100]}")
            break
    else:
        print("\n⚠ error_sequences.txt not found")
        print("  This means either:")
        print("    1. Evaluation hasn't run")
        print("    2. Model predicts nothing (file was empty and deleted)")
    
    print("\n" + "="*60)
    print("Recommendations:")
    print("="*60)
    print("1. To manually run evaluation, you need to:")
    print("   - Load the model with the checkpoint")
    print("   - Run the evaluate() function from train_MD-FED.py")
    print("   - Check the coarse_scores distribution")
    print()
    print("2. If model predicts all class 0:")
    print("   - Check data balance (too many 'no event' frames?)")
    print("   - Try class-weighted loss")
    print("   - Lower learning rate")
    print("   - Check if skeleton features are meaningful")
    print()
    print("3. Check training logs for evaluation output:")
    print("   - Look for 'Mean F1' messages")
    print("   - Check if evaluation ran at all")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_model_predictions.py <checkpoint_path> [output_dir]")
        print("Example: python check_model_predictions.py MD-FED/md_fed_outputs/stage1/checkpoint_049.pt MD-FED/md_fed_outputs/stage1")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(checkpoint_path)
    
    check_model_outputs(checkpoint_path, output_dir)
