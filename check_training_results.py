#!/usr/bin/env python3
"""
Check MD-FED Stage 1 training results
"""

import os
import json
import argparse
import numpy as np


def check_training_results(output_dir):
    """Check training results and provide diagnostics"""
    
    loss_file = os.path.join(output_dir, 'loss.json')
    config_file = os.path.join(output_dir, 'config.json')
    
    print("="*60)
    print("Training Results Check")
    print("="*60)
    
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
    import glob
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
            print("  1. Validation data has no labels or incorrect format")
            print("  2. Model predictions are all background (coarse_pred = 0)")
            print("  3. start_val_epoch was too high (validation never ran)")
            print("  4. Evaluation function issue")
            print("\n  Recommendations:")
            print("  - Check if val.json has correct event labels")
            print("  - Check if skeleton pkl files match video names in val.json")
            print("  - Try using 'loss' criterion instead of 'edit'")
            print("  - Check error_sequences.txt in MD-FED directory for details")
        else:
            print("\n✓ Validation was performed")
            best_edit = max(val_edits)
            if best_edit > 0:
                print(f"  Best edit score: {best_edit:.5f}")
            else:
                print("  But all edit scores are 0 - model may not be learning event detection")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Training completed - checkpoints are saved")
    print("2. For Stage 2, you can use the latest checkpoint or best loss checkpoint")
    print("3. If scores are 0, check validation data format and labels")
    print("4. Stage 1 focuses on skeleton feature learning - low scores may be normal")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Check MD-FED training results')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Training output directory (e.g., md_fed_outputs/stage1)'
    )
    
    args = parser.parse_args()
    check_training_results(args.output_dir)


if __name__ == '__main__':
    main()
