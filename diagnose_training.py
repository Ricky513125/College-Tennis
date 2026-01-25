#!/usr/bin/env python3
"""
Diagnostic script to check training issues:
1. Check if model is predicting any events
2. Check prediction distributions
3. Check data loading
"""

import os
import sys
import json
import numpy as np

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
            # Check start_val_epoch
            num_epochs = len(losses)
            start_val_epoch = num_epochs - 20  # BASE_NUM_VAL_EPOCHS = 20
            print(f"  Expected start_val_epoch: {start_val_epoch}")
            print(f"  This means evaluation should run from epoch {start_val_epoch} onwards")
            if num_epochs >= start_val_epoch:
                print(f"  ⚠ Evaluation should have run but didn't - check evaluation code")
        
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
        print(f"  Directory contents:")
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                print(f"    - {f}")
        return
    
    # Check error sequences file - check multiple possible locations
    error_files = [
        'error_sequences.txt',  # Current directory
        os.path.join(output_dir, 'error_sequences.txt'),  # Output directory
        os.path.join(os.path.dirname(output_dir), 'error_sequences.txt'),  # Parent directory
        'MD-FED/error_sequences.txt',  # MD-FED directory (if running from project root)
    ]
    
    error_file_found = None
    for error_file in error_files:
        if os.path.exists(error_file):
            error_file_found = error_file
            break
    
    if error_file_found:
        with open(error_file_found, 'r') as f:
            content = f.read()
        if len(content.strip()) == 0:
            print(f"\n⚠ {error_file_found} is EMPTY")
            print("  This strongly suggests the model is predicting NO EVENTS at all")
            print("  Possible causes:")
            print("    1. Model always predicts class 0 (no event)")
            print("    2. coarse_scores[:, 0] > coarse_scores[:, 1] for all frames")
            print("    3. Model hasn't learned to detect events")
        else:
            lines = content.strip().split('\n')
            video_count = len([l for l in lines if 'video' in l.lower() or l.endswith('.mp4') or l.endswith('.avi') or '/' in l])
            print(f"\n✓ Found {video_count} error sequences in {error_file_found}")
            print("  This means the model IS making predictions, but they're wrong")
            print(f"  First few lines:")
            for line in lines[:10]:
                if line.strip():
                    print(f"    {line[:80]}")
    else:
        print(f"\n⚠ error_sequences.txt not found in any expected location")
        print("  Checked locations:")
        for ef in error_files:
            print(f"    - {ef}")
        print("  This could mean:")
        print("    1. Evaluation hasn't run yet")
        print("    2. Model predicts nothing (file created but empty, then deleted?)")
        print("    3. File is in a different location")
    
    # Check checkpoints
    checkpoint_files = []
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith('checkpoint_') and f.endswith('.pt'):
                checkpoint_files.append(f)
    
    if checkpoint_files:
        print(f"\n✓ Found {len(checkpoint_files)} checkpoint files")
        epochs = []
        for f in checkpoint_files:
            try:
                epoch = int(f.replace('checkpoint_', '').replace('.pt', ''))
                epochs.append(epoch)
            except:
                pass
        if epochs:
            print(f"  Epochs: {min(epochs)} to {max(epochs)}")
    else:
        print(f"\n⚠ No checkpoint files found in {output_dir}")
    
    print("\n" + "="*60)
    print("Diagnosis Summary:")
    print("="*60)
    
    # Final diagnosis
    if error_file_found:
        with open(error_file_found, 'r') as f:
            content = f.read()
        if len(content.strip()) == 0:
            print("\n🔴 CRITICAL: Model is not predicting any events")
            print("\nRoot cause: Model always predicts class 0 (no event)")
            print("\nRecommended fixes:")
            print("  1. Check data balance:")
            print("     - Are there enough 'event' frames vs 'no event' frames?")
            print("     - Try class weighting in loss function")
            print("  2. Lower learning rate:")
            print("     python run_md_fed_stage1.py --learning_rate 0.0001 ...")
            print("  3. Check skeleton data:")
            print("     - Verify pose_dir contains valid .pkl files")
            print("     - Check if skeleton features are being extracted correctly")
            print("  4. Check labels:")
            print("     - Verify train.json and val.json have correct event labels")
            print("  5. Try different initialization or longer training")
        else:
            print("\n🟡 PARTIAL: Model is making predictions but they're incorrect")
            print("\nThis is progress! The model is learning but needs:")
            print("  1. More training epochs")
            print("  2. Better hyperparameters")
            print("  3. Data augmentation")
    else:
        print("\n⚠️  UNCERTAIN: Cannot determine prediction status")
        print("\nNext steps:")
        print("  1. Check if evaluation ran (epoch >= start_val_epoch)")
        print("  2. Manually run evaluation on a checkpoint")
        print("  3. Check training logs for evaluation output")
    
    if val_edits and all(e == 0 for e in val_edits):
        print("\n⚠️  Evaluation may not have run properly")
        print("   Check that:")
        print("   - epoch >= start_val_epoch (should be epoch 30+ for 50 epochs)")
        print("   - val_data_frames is not None")
        print("   - Evaluation function completed without errors")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_training.py <output_dir>")
        print("Example: python diagnose_training.py MD-FED/md_fed_outputs/stage1")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    check_predictions(output_dir)
