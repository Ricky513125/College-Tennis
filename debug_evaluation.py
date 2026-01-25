#!/usr/bin/env python3
"""
Add debug output to evaluation to see why F1 is 0.
This will show prediction vs label alignment.
"""

import sys
import os

# Add MD-FED to path
sys.path.insert(0, 'MD-FED')

def add_debug_to_evaluate():
    """Add debug output to evaluate function"""
    
    eval_file = 'MD-FED/train_MD-FED.py'
    
    # Read the file
    with open(eval_file, 'r') as f:
        content = f.read()
    
    # Find the evaluation section and add debug output
    # After line 440: coarse_label, fine_label = dataset.get_labels(video)
    
    old_code = """    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        coarse_label, fine_label = dataset.get_labels(video)
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        # argmax pred
        coarse_pred = np.argmax(coarse_scores, axis=1)"""
    
    new_code = """    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        coarse_label, fine_label = dataset.get_labels(video)
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        # DEBUG: Print first video's labels and predictions
        if video == sorted(pred_dict.keys())[0]:
            print(f"\\nDEBUG: First video: {video}")
            print(f"  Video length: {len(coarse_label)}")
            print(f"  Coarse label events: {np.sum(coarse_label)}")
            print(f"  Coarse label positions: {np.where(coarse_label == 1)[0][:10]}")
            print(f"  Coarse scores shape: {coarse_scores.shape}")
            print(f"  Coarse scores (first 10, class 0): {coarse_scores[:10, 0]}")
            print(f"  Coarse scores (first 10, class 1): {coarse_scores[:10, 1]}")

        # argmax pred
        coarse_pred = np.argmax(coarse_scores, axis=1)
        
        # DEBUG: Print predictions for first video
        if video == sorted(pred_dict.keys())[0]:
            print(f"  Coarse pred events: {np.sum(coarse_pred)}")
            print(f"  Coarse pred positions: {np.where(coarse_pred == 1)[0][:10]}")"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back
        with open(eval_file, 'w') as f:
            f.write(content)
        
        print("✓ Added debug output to evaluate function")
        print("  Run evaluation again to see prediction vs label alignment")
    else:
        print("⚠ Could not find exact code to replace")
        print("  You may need to manually add debug output")


if __name__ == '__main__':
    add_debug_to_evaluate()
