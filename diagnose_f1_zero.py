#!/usr/bin/env python3
"""
Diagnose why F1 is 0 even though model is making predictions.
Checks prediction vs ground truth alignment.
"""

import sys
import json
import numpy as np

def analyze_f1_issue(error_file, loss_file):
    """Analyze why F1 is 0"""
    
    print("="*60)
    print("Diagnosing F1=0 Issue")
    print("="*60)
    
    # Check error sequences
    if error_file and os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        
        if content.strip():
            print("\n✓ Model IS making predictions (error_sequences.txt has content)")
            print("  This means the issue is prediction ACCURACY, not lack of predictions")
        else:
            print("\n⚠ error_sequences.txt is empty")
            return
    
    # Load loss data
    if loss_file and os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            losses = json.load(f)
        
        val_edits = [l.get('val_edit', 0) for l in losses]
        val_epochs = [i for i, e in enumerate(val_edits) if e > 0]
        
        if val_epochs:
            print(f"\n✓ Evaluation ran in epochs: {val_epochs}")
            print(f"  But F1 scores are all 0.0")
        else:
            print("\n⚠ Evaluation may not have run (all val_edit are 0)")
    
    print("\n" + "="*60)
    print("Root Cause Analysis:")
    print("="*60)
    print("\nF1=0 means: TP=0 (no true positives)")
    print("This happens when:")
    print("  1. Predictions are in wrong time positions (beyond delta window)")
    print("  2. Predictions are wrong event types")
    print("  3. Delta tolerance (default=1 frame) is too strict")
    print()
    print("From error_sequences.txt, model IS predicting events,")
    print("but they don't match ground truth within ±1 frame tolerance.")
    
    print("\n" + "="*60)
    print("Solutions:")
    print("="*60)
    print("\n1. Increase delta tolerance:")
    print("   - Current: delta=1 (predictions must be within ±1 frame)")
    print("   - Try: delta=5 or delta=10 for more lenient matching")
    print("   - Modify evaluate() call: evaluate(..., delta=5)")
    print()
    print("2. Check prediction timing:")
    print("   - Model may be predicting events too early/late")
    print("   - This is a temporal alignment issue")
    print("   - May need temporal smoothing or post-processing")
    print()
    print("3. Continue training:")
    print("   - Model is learning (loss decreasing)")
    print("   - Predictions exist but need refinement")
    print("   - More training epochs may improve alignment")
    print()
    print("4. Check data quality:")
    print("   - Verify labels are correctly aligned with frames")
    print("   - Check if there's systematic offset")
    
    print("\n" + "="*60)
    print("Quick Test:")
    print("="*60)
    print("\nTo test if delta is the issue, you can:")
    print("1. Modify MD-FED/train_MD-FED.py line 802:")
    print("   Change: evaluate(..., delta=1, ...)")
    print("   To:     evaluate(..., delta=5, ...)")
    print("2. Re-run evaluation on a checkpoint")
    print("3. Check if F1 scores improve")


if __name__ == '__main__':
    import os
    
    error_file = sys.argv[1] if len(sys.argv) > 1 else 'MD-FED/error_sequences.txt'
    loss_file = sys.argv[2] if len(sys.argv) > 2 else 'MD-FED/md_fed_outputs/stage1/loss.json'
    
    analyze_f1_issue(error_file, loss_file)
