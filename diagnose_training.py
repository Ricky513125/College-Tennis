#!/usr/bin/env python3
"""
Diagnostic script to check training issues:
1. Check if model is predicting any events
2. Check prediction distributions
3. Check data loading
"""

import os
import sys
import numpy as np
import torch
import json

def check_predictions(output_dir):
    """Check what the model is actually predicting"""
    print("="*60)
    print("Diagnosing Training Issues")
    print("="*60)
    
    # Check loss file
    loss_file = os.path.join(output_dir, 'loss.json')
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
            print(f"\n⚠ Train loss is NOT decreasing - model may not be learning properly")
        
        # Check validation epochs
        val_epochs = [i for i, e in enumerate(val_edits) if e > 0]
        if val_epochs:
            print(f"\n✓ Validation performed in epochs: {val_epochs}")
            print(f"  Best val_edit: {max([val_edits[i] for i in val_epochs]):.5f}")
        else:
            print(f"\n⚠ No validation performed (all val_edit are 0)")
            print(f"  This is normal if start_val_epoch was not reached")
        
        # Find best epoch by loss
        best_loss_epoch = np.argmin(val_losses)
        print(f"\nBest epoch by validation loss: {best_loss_epoch} (loss: {val_losses[best_loss_epoch]:.5f})")
        
        # Check if validation loss is consistently higher
        val_higher_count = sum(1 for i in range(len(train_losses)) if val_losses[i] > train_losses[i])
        print(f"\nValidation loss higher than training loss in {val_higher_count}/{len(train_losses)} epochs")
        if val_higher_count > len(train_losses) * 0.7:
            print("  ⚠ This suggests possible overfitting or data distribution mismatch")
        
    else:
        print(f"\n✗ loss.json not found in {output_dir}")
    
    # Check error sequences file
    error_file = 'error_sequences.txt'
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        if len(content.strip()) == 0:
            print(f"\n⚠ {error_file} is empty - this could mean:")
            print("  1. Model is predicting no events at all")
            print("  2. Model predictions are all correct (unlikely with F1=0)")
        else:
            lines = content.strip().split('\n')
            print(f"\n✓ Found {len([l for l in lines if l.startswith('video')])} error sequences in {error_file}")
    else:
        print(f"\n⚠ {error_file} not found - evaluation may not have run properly")
    
    print("\n" + "="*60)
    print("Recommendations:")
    print("="*60)
    print("1. Check if model is learning:")
    print("   - If train loss is not decreasing, try:")
    print("     * Lower learning rate (e.g., 0.0001)")
    print("     * Check data loading (are labels correct?)")
    print("     * Check if model architecture is correct for Stage 1")
    print()
    print("2. Check predictions:")
    print("   - Model may be predicting all 'no event' (class 0)")
    print("   - This would cause F1=0 and Edit score=0")
    print("   - Check coarse_scores distribution in evaluation")
    print()
    print("3. Validation loss > Training loss:")
    print("   - This can be normal, especially early in training")
    print("   - But combined with F1=0, suggests model isn't learning")
    print()
    print("4. For Stage 1 training:")
    print("   - Make sure skeleton data is being loaded correctly")
    print("   - Check that pose_dir contains valid .pkl files")
    print("   - Verify data preparation was done correctly")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_training.py <output_dir>")
        print("Example: python diagnose_training.py md_fed_outputs/stage1")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    check_predictions(output_dir)
