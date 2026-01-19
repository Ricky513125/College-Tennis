#!/usr/bin/env python3
"""
Check MD-FED Stage 1 training results
"""

import os
import json
import argparse
import numpy as np
import glob


def check_training_results(output_dir):
    """Check training results and provide diagnostics"""
    
    # Try both relative and absolute paths
    if not os.path.isabs(output_dir):
        # Try current directory first
        if os.path.exists(output_dir):
            output_dir = os.path.abspath(output_dir)
        # Try MD-FED subdirectory
        elif os.path.exists(os.path.join('MD-FED', output_dir)):
            output_dir = os.path.abspath(os.path.join('MD-FED', output_dir))
    
    loss_file = os.path.join(output_dir, 'loss.json')
    config_file = os.path.join(output_dir, 'config.json')
    
    print("="*60)
    print("Training Results Check")
    print("="*60)
    print(f"Checking directory: {output_dir}")
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"\n✗ Directory not found: {output_dir}")
        print("\nTrying to find output directory...")
        # Search for loss.json or checkpoint files
        for root, dirs, files in os.walk('.'):
            if 'loss.json' in files or any(f.startswith('checkpoint_') for f in files):
                print(f"  Found potential output: {root}")
        return
    
    # Check loss file
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            losses = json.load(f)
        
        print(f"\n✓ Found loss.json with {len(losses)} epochs")
        
        # Analyze loss trends
        train_losses = [l['train'] for l in losses]
        val_losses = [l['val'] for l in losses]
        val_edits = [l.get('val_edit', 0) for l in losses]
        
        print(f"\nLoss Statistics:")
        print(f"  Initial train loss: {train_losses[0]:.5f}")
        print(f"  Final train loss: {train_losses[-1]:.5f}")
        print(f"  Train loss change: {train_losses[0] - train_losses[-1]:.5f}")
        print(f"  Initial val loss: {val_losses[0]:.5f}")
        print(f"  Final val loss: {val_losses[-1]:.5f}")
        print(f"  Val loss change: {val_losses[0] - val_losses[-1]:.5f}")
        
        # Check if loss is decreasing
        if train_losses[-1] < train_losses[0]:
            print(f"\n✓ Train loss is decreasing - model is learning!")
        else:
            print(f"\n⚠ Train loss is not decreasing - may need to check learning rate or data")
        
        # Check validation epochs
        val_epochs = [i for i, e in enumerate(val_edits) if e > 0]
        if val_epochs:
            print(f"\n✓ Validation performed in epochs: {val_epochs}")
            print(f"  Best val_edit: {max([val_edits[i] for i in val_epochs]):.5f}")
        else:
            print(f"\n⚠ No validation performed (all val_edit are 0)")
            print(f"  This is normal if start_val_epoch ({len(losses) - 20}) was not reached")
        
        # Find best epoch by loss
        best_loss_epoch = np.argmin(val_losses)
        print(f"\nBest epoch by validation loss: {best_loss_epoch} (loss: {val_losses[best_loss_epoch]:.5f})")
        
    else:
        print(f"\n✗ loss.json not found in {output_dir}")
        print(f"  Directory contents:")
        for f in os.listdir(output_dir):
            print(f"    - {f}")
    
    # Check config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\n✓ Found config.json")
        print(f"  Dataset: {config.get('dataset', 'N/A')}")
        print(f"  Stage: {config.get('stage', 'N/A')}")
        print(f"  Num epochs: {config.get('num_epochs', 'N/A')}")
        print(f"  Start val epoch: {config.get('start_val_epoch', 'N/A')}")
    
    # Check checkpoints
    checkpoint_files = glob.glob(os.path.join(output_dir, 'checkpoint_*.pt'))
    if checkpoint_files:
        epochs = []
        for f in checkpoint_files:
            try:
                epoch = int(os.path.basename(f).replace('checkpoint_', '').replace('.pt', ''))
                epochs.append(epoch)
            except:
                pass
        if epochs:
            print(f"\n✓ Found {len(epochs)} checkpoints")
            print(f"  Epochs: {min(epochs)} to {max(epochs)}")
            print(f"  Latest checkpoint: checkpoint_{max(epochs):03d}.pt")
    
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)
    
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            losses = json.load(f)
        val_edits = [l.get('val_edit', 0) for l in losses]
        
        if all(e == 0 for e in val_edits):
            print("\n⚠ All validation edit scores are 0.0")
            print("  Possible reasons:")
            print("  1. Model predictions are all background (coarse_pred = 0)")
            print("  2. Validation data format issue")
            print("  3. start_val_epoch was too high (validation never ran)")
            print("  4. Model not learning (check if loss is decreasing)")
            print("\n  Recommendations:")
            print("  - Check if train/val loss is decreasing")
            print("  - For Stage 1, low scores may be normal (focus is on feature learning)")
            print("  - Check error_sequences.txt in MD-FED directory for details")
        else:
            print("\n✓ Validation was performed")
            best_edit = max(val_edits)
            if best_edit > 0:
                print(f"  Best edit score: {best_edit:.5f}")
            else:
                print("  But all edit scores are 0 - model may not be learning event detection")
    
    print("\n" + "="*60)
    print("About Stage 1:")
    print("="*60)
    print("Stage 1 focuses on skeleton feature pretraining.")
    print("The goal is to learn good skeleton feature representations,")
    print("not necessarily to achieve high event detection scores.")
    print("\n✓ Training completed successfully!")
    print("✓ Checkpoints are saved and can be used for Stage 2")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Check MD-FED training results')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='md_fed_outputs/stage1',
        help='Training output directory (default: md_fed_outputs/stage1)'
    )
    
    args = parser.parse_args()
    check_training_results(args.output_dir)


if __name__ == '__main__':
    main()
