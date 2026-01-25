#!/usr/bin/env python3
"""
Analyze error_sequences.txt to understand prediction patterns
"""

import sys
import re
from collections import Counter

def analyze_errors(error_file):
    """Analyze error sequences to understand prediction patterns"""
    
    print("="*60)
    print("Analyzing Error Sequences")
    print("="*60)
    
    with open(error_file, 'r') as f:
        content = f.read()
    
    if not content.strip():
        print("File is empty - model predicts no events")
        return
    
    # Parse error sequences
    errors = []
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == '------------------------':
            i += 1
            continue
        
        # Video name
        if '/' in line or line.endswith('.mp4') or '_' in line:
            video = line
            i += 1
            
            # Prediction sequence
            pred_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            # Skip empty line
            if i < len(lines) and not lines[i].strip():
                i += 1
            
            # Ground truth sequence
            gt_line = lines[i].strip() if i < len(lines) else ""
            i += 1
            
            errors.append({
                'video': video,
                'pred': pred_line,
                'gt': gt_line
            })
        else:
            i += 1
    
    print(f"\n✓ Found {len(errors)} error sequences")
    
    # Analyze patterns
    pred_empty = sum(1 for e in errors if not e['pred'])
    gt_empty = sum(1 for e in errors if not e['gt'])
    both_have = sum(1 for e in errors if e['pred'] and e['gt'])
    
    print(f"\nPattern Analysis:")
    print(f"  Predictions empty: {pred_empty}")
    print(f"  Ground truth empty: {gt_empty}")
    print(f"  Both have content: {both_have}")
    
    # Count prediction types
    pred_events = []
    gt_events = []
    
    for e in errors:
        if e['pred']:
            pred_events.extend(e['pred'].split('->'))
        if e['gt']:
            gt_events.extend(e['gt'].split('->'))
    
    pred_counter = Counter(pred_events)
    gt_counter = Counter(gt_events)
    
    print(f"\nMost common predicted events:")
    for event, count in pred_counter.most_common(10):
        print(f"  {event}: {count}")
    
    print(f"\nMost common ground truth events:")
    for event, count in gt_counter.most_common(10):
        print(f"  {event}: {count}")
    
    # Check if model is over-predicting
    total_pred_events = len(pred_events)
    total_gt_events = len(gt_events)
    
    print(f"\nEvent Counts:")
    print(f"  Total predicted events: {total_pred_events}")
    print(f"  Total ground truth events: {total_gt_events}")
    
    if total_pred_events > total_gt_events * 2:
        print("  ⚠ Model is OVER-PREDICTING (predicting too many events)")
    elif total_pred_events < total_gt_events * 0.5:
        print("  ⚠ Model is UNDER-PREDICTING (missing many events)")
    else:
        print("  ✓ Event counts are similar")
    
    # Show some examples
    print(f"\nExample errors (first 5):")
    for i, e in enumerate(errors[:5]):
        print(f"\n  {i+1}. Video: {e['video'][:60]}...")
        print(f"     Predicted: {e['pred'] or '(empty)'}")
        print(f"     Ground truth: {e['gt'] or '(empty)'}")
    
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)
    
    if gt_empty > len(errors) * 0.5:
        print("\n⚠ Many ground truth sequences are empty")
        print("  This could mean:")
        print("  1. Model is predicting events where there are none (false positives)")
        print("  2. These videos have no labeled events")
        print("  3. Label format issue")
    
    if pred_empty > len(errors) * 0.5:
        print("\n⚠ Many predictions are empty")
        print("  This means model is missing events (false negatives)")
    
    if both_have > 0:
        print(f"\n✓ Model is making predictions ({both_have} sequences with both pred and GT)")
        print("  The issue is prediction accuracy, not lack of predictions")
        print("  This is actually good progress - model is learning but needs refinement")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_error_sequences.py <error_sequences.txt>")
        print("Example: python analyze_error_sequences.py MD-FED/error_sequences.txt")
        sys.exit(1)
    
    error_file = sys.argv[1]
    analyze_errors(error_file)
