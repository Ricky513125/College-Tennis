#!/usr/bin/env python3
"""
Check what the model is actually predicting during evaluation.
This helps diagnose why F1 scores are 0.
"""

import os
import sys
import numpy as np
import json

def analyze_predictions(output_dir):
    """Analyze prediction patterns from loss.json and error_sequences.txt"""
    
    print("="*60)
    print("Prediction Analysis")
    print("="*60)
    
    # Check loss trends
    loss_file = os.path.join(output_dir, 'loss.json')
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            losses = json.load(f)
        
        print(f"\n✓ Analyzing {len(losses)} epochs")
        
        # Check if loss is decreasing
        train_losses = [l['train'] for l in losses]
        val_losses = [l['val'] for l in losses]
        
        initial_train = train_losses[0]
        final_train = train_losses[-1]
        initial_val = val_losses[0]
        final_val = val_losses[-1]
        
        print(f"\nLoss Trends:")
        print(f"  Train: {initial_train:.5f} -> {final_train:.5f} (change: {initial_train - final_train:+.5f})")
        print(f"  Val:   {initial_val:.5f} -> {final_val:.5f} (change: {initial_val - final_val:+.5f})")
        
        if final_train < initial_train:
            print("  ✓ Train loss is decreasing")
        else:
            print("  ⚠ Train loss is NOT decreasing - model may not be learning")
        
        # Check loss stability
        recent_train = train_losses[-10:] if len(train_losses) >= 10 else train_losses
        train_std = np.std(recent_train)
        print(f"\n  Recent train loss std: {train_std:.5f}")
        if train_std < 0.001:
            print("  ⚠ Loss is very stable - model may have converged to a bad solution")
        
    else:
        print(f"\n✗ loss.json not found")
        return
    
    # Check error sequences
    error_file = 'error_sequences.txt'
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        
        if len(content.strip()) == 0:
            print(f"\n⚠ {error_file} is EMPTY")
            print("  This strongly suggests the model is predicting NO EVENTS at all")
            print("  Possible causes:")
            print("    1. Model always predicts class 0 (no event)")
            print("    2. coarse_scores[:, 0] > coarse_scores[:, 1] for all frames")
            print("    3. Model hasn't learned to detect events")
        else:
            lines = content.strip().split('\n')
            video_count = len([l for l in lines if 'video' in l.lower() or l.endswith('.mp4') or l.endswith('.avi')])
            print(f"\n✓ Found {video_count} error sequences")
            print("  This means the model IS making predictions, but they're wrong")
    else:
        print(f"\n⚠ {error_file} not found")
        print("  This could mean evaluation hasn't run, or model predicts nothing")
    
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)
    
    # Final diagnosis
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        if len(content.strip()) == 0:
            print("\n🔴 PROBLEM: Model is not predicting any events")
            print("\nRecommended fixes:")
            print("  1. Check if skeleton data is being loaded correctly")
            print("  2. Verify data preparation (run prepare_md_fed_data.py)")
            print("  3. Try lower learning rate (0.0001 or 0.0005)")
            print("  4. Check if model architecture is correct for Stage 1")
            print("  5. Verify labels in train.json and val.json are correct")
            print("  6. Check if there's class imbalance (too many 'no event' frames)")
        else:
            print("\n🟡 PARTIAL: Model is making predictions but they're incorrect")
            print("\nThis is actually progress! The model is learning but needs:")
            print("  1. More training epochs")
            print("  2. Better hyperparameters")
            print("  3. Data augmentation")
    else:
        print("\n⚠️  Cannot determine - error_sequences.txt not found")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_predictions.py <output_dir>")
        print("Example: python check_predictions.py md_fed_outputs/stage1")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    analyze_predictions(output_dir)
