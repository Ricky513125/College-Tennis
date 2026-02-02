#!/usr/bin/env python3
"""
检查训练状态，显示训练历史和最佳模型路径
"""

import os
import sys
import json
import argparse

# Add MD-FED to path
md_fed_dir = os.path.join(os.path.dirname(__file__), 'MD-FED')
if os.path.exists(md_fed_dir):
    sys.path.insert(0, md_fed_dir)

from util.io import load_json
from train_MD_FED import get_best_epoch_and_history, get_last_epoch


def check_training_status(save_dir):
    """检查训练状态"""
    print(f'\n{"="*60}')
    print(f'Training Status Check')
    print(f'{"="*60}\n')
    
    if not os.path.exists(save_dir):
        print(f"❌ Directory not found: {save_dir}")
        return
    
    print(f"📁 Directory: {os.path.abspath(save_dir)}\n")
    
    # 检查 loss.json
    loss_file = os.path.join(save_dir, 'loss.json')
    if os.path.exists(loss_file):
        losses = load_json(loss_file)
        if losses:
            last_epoch = max(e['epoch'] for e in losses)
            best_entry = max(losses, key=lambda x: x.get('val_edit', 0))
            best_epoch = best_entry['epoch']
            best_edit = best_entry.get('val_edit', 0)
            
            print(f"📊 Training History:")
            print(f"   Total epochs: {len(losses)}")
            print(f"   Last epoch: {last_epoch}")
            print(f"   Best epoch: {best_epoch}")
            print(f"   Best Edit score: {best_edit:.4f}")
            
            # 显示最近几个 epoch
            print(f"\n📈 Recent epochs (last 10):")
            for entry in losses[-10:]:
                epoch = entry['epoch']
                train_loss = entry.get('train', 0)
                val_loss = entry.get('val', 0)
                val_edit = entry.get('val_edit', 0)
                marker = " ⭐" if epoch == best_epoch else ""
                print(f"   Epoch {epoch:3d}: train={train_loss:.5f}, val={val_loss:.5f}, edit={val_edit:.4f}{marker}")
        else:
            print("⚠️  loss.json is empty")
    else:
        print("⚠️  loss.json not found")
    
    # 检查检查点文件
    print(f"\n📂 Checkpoint Files:")
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_')]
    if checkpoint_files:
        epochs = sorted([int(f.replace('checkpoint_', '').replace('.pt', '')) for f in checkpoint_files])
        print(f"   Found {len(epochs)} checkpoints")
        print(f"   Epoch range: {epochs[0]} to {epochs[-1]}")
        
        # 检查最佳检查点
        if losses and best_epoch in epochs:
            best_checkpoint = os.path.join(save_dir, f'checkpoint_{best_epoch:03d}.pt')
            if os.path.exists(best_checkpoint):
                file_size = os.path.getsize(best_checkpoint) / (1024 * 1024)  # MB
                print(f"\n   ⭐ Best checkpoint:")
                print(f"      Path: {os.path.abspath(best_checkpoint)}")
                print(f"      Epoch: {best_epoch}")
                print(f"      Edit score: {best_edit:.4f}")
                print(f"      Size: {file_size:.2f} MB")
                print(f"      ✓ File exists")
            else:
                print(f"\n   ⚠️  Best checkpoint file not found: {best_checkpoint}")
        
        # 检查最后检查点
        if epochs:
            last_checkpoint = os.path.join(save_dir, f'checkpoint_{epochs[-1]:03d}.pt')
            if os.path.exists(last_checkpoint):
                file_size = os.path.getsize(last_checkpoint) / (1024 * 1024)  # MB
                print(f"\n   📌 Last checkpoint:")
                print(f"      Path: {os.path.abspath(last_checkpoint)}")
                print(f"      Epoch: {epochs[-1]}")
                print(f"      Size: {file_size:.2f} MB")
                print(f"      ✓ File exists")
    else:
        print("   ⚠️  No checkpoint files found")
    
    # 检查配置文件
    config_file = os.path.join(save_dir, 'config.json')
    if os.path.exists(config_file):
        config = load_json(config_file)
        print(f"\n⚙️  Configuration:")
        print(f"   Dataset: {config.get('dataset', 'unknown')}")
        print(f"   Visual arch: {config.get('visual_arch', 'unknown')}")
        print(f"   Learning rate: {config.get('learning_rate', 'unknown')}")
        print(f"   Batch size: {config.get('batch_size', 'unknown')}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check training status')
    parser.add_argument('save_dir', type=str, help='Training save directory')
    args = parser.parse_args()
    
    check_training_status(args.save_dir)
