#!/usr/bin/env python3
"""
检查训练数据的平衡性
"""

import json
import sys
import numpy as np

def check_balance(json_file):
    """检查数据平衡"""
    
    print("="*60)
    print(f"检查数据平衡: {json_file}")
    print("="*60)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_videos = len(data)
    total_frames = 0
    total_events = 0
    videos_with_events = 0
    videos_without_events = 0
    
    event_frame_count = 0
    no_event_frame_count = 0
    
    for video_data in data:
        num_frames = video_data.get('num_frames', 0)
        events = video_data.get('events', [])
        num_events = len(events)
        
        total_frames += num_frames
        total_events += num_events
        
        if num_events > 0:
            videos_with_events += 1
            # 每个事件占用1帧（假设）
            event_frame_count += num_events
            no_event_frame_count += (num_frames - num_events)
        else:
            videos_without_events += 1
            no_event_frame_count += num_frames
    
    print(f"\n视频统计:")
    print(f"  总视频数: {total_videos}")
    print(f"  有事件的视频: {videos_with_events}")
    print(f"  无事件的视频: {videos_without_events}")
    
    print(f"\n帧统计:")
    print(f"  总帧数: {total_frames}")
    print(f"  事件帧数: {event_frame_count}")
    print(f"  无事件帧数: {no_event_frame_count}")
    
    if total_frames > 0:
        event_ratio = event_frame_count / total_frames * 100
        no_event_ratio = no_event_frame_count / total_frames * 100
        
        print(f"\n比例:")
        print(f"  事件帧: {event_ratio:.2f}%")
        print(f"  无事件帧: {no_event_ratio:.2f}%")
        
        imbalance_ratio = no_event_frame_count / event_frame_count if event_frame_count > 0 else float('inf')
        print(f"\n不平衡比例: {imbalance_ratio:.1f}:1 (无事件:有事件)")
        
        print("\n" + "="*60)
        print("诊断:")
        print("="*60)
        
        if event_ratio < 1:
            print("\n🔴 严重不平衡！事件帧 < 1%")
            print("   模型会倾向于总是预测'无事件'")
            print("   强烈建议使用加权损失函数")
        elif event_ratio < 5:
            print("\n⚠️  不平衡：事件帧 < 5%")
            print("   建议使用加权损失函数")
        elif event_ratio < 10:
            print("\n⚠️  轻微不平衡：事件帧 < 10%")
            print("   可以考虑使用加权损失函数")
        else:
            print("\n✓ 数据相对平衡")
        
        if imbalance_ratio > 100:
            print(f"\n⚠️  不平衡比例 {imbalance_ratio:.1f}:1 非常高")
            print("   需要给事件类别至少 {:.0f} 倍权重".format(imbalance_ratio / 10))
    
    print(f"\n事件统计:")
    print(f"  总事件数: {total_events}")
    print(f"  平均每视频事件数: {total_events / total_videos:.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        json_file = 'md_fed_data/f3set-tennis-sub/train.json'
    else:
        json_file = sys.argv[1]
    
    check_balance(json_file)
