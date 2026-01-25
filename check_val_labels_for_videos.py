#!/usr/bin/env python3
"""
检查验证集中特定视频是否有标签
"""

import json
import sys

def check_videos_in_val(video_names, val_json_file):
    """检查这些视频在验证集中是否有标签"""
    
    print("="*60)
    print(f"检查验证集中的视频标签")
    print(f"验证集文件: {val_json_file}")
    print("="*60)
    
    # 加载验证集
    with open(val_json_file, 'r') as f:
        val_data = json.load(f)
    
    print(f"\n验证集总视频数: {len(val_data)}")
    
    # 创建视频名到数据的映射
    video_dict = {v['video']: v for v in val_data}
    
    # 检查每个视频
    found_videos = []
    missing_videos = []
    
    for video_name in video_names:
        # 尝试匹配（可能视频名格式不同）
        matched = False
        for val_video_name, val_video_data in video_dict.items():
            # 检查是否包含关键部分
            if video_name in val_video_name or val_video_name in video_name:
                matched = True
                found_videos.append({
                    'query': video_name,
                    'matched': val_video_name,
                    'data': val_video_data
                })
                break
        
        if not matched:
            missing_videos.append(video_name)
    
    print(f"\n找到匹配: {len(found_videos)}")
    print(f"未找到: {len(missing_videos)}")
    
    # 检查找到的视频的标签
    print("\n" + "="*60)
    print("找到的视频标签信息:")
    print("="*60)
    
    for i, item in enumerate(found_videos[:5]):
        video_data = item['data']
        events = video_data.get('events', [])
        
        print(f"\n视频 {i+1}:")
        print(f"  查询名: {item['query'][:60]}...")
        print(f"  匹配名: {item['matched'][:60]}...")
        print(f"  总帧数: {video_data.get('num_frames', 0)}")
        print(f"  事件数: {len(events)}")
        
        if events:
            print(f"  前 3 个事件:")
            for j, event in enumerate(events[:3]):
                print(f"    Frame {event.get('frame', -1)}: {event.get('label', 'unknown')}")
        else:
            print(f"  ⚠️  没有事件标签！")
    
    if missing_videos:
        print("\n" + "="*60)
        print("未找到的视频:")
        print("="*60)
        for video in missing_videos[:5]:
            print(f"  {video[:60]}...")
    
    print("\n" + "="*60)
    print("诊断:")
    print("="*60)
    
    if len(found_videos) > 0:
        videos_without_events = sum(1 for item in found_videos if len(item['data'].get('events', [])) == 0)
        if videos_without_events > 0:
            print(f"\n⚠️  找到 {videos_without_events} 个视频但没有事件标签")
            print("  这解释了为什么 error_sequences.txt 中真实标签序列为空")
    
    if len(missing_videos) > 0:
        print(f"\n⚠️  有 {len(missing_videos)} 个视频在验证集中找不到")
        print("  可能原因:")
        print("  1. 视频名格式不匹配")
        print("  2. 这些视频不在验证集中")
        print("  3. 验证集文件不正确")


if __name__ == '__main__':
    # 从 error_sequences.txt 提取视频名
    error_file = 'MD-FED/error_sequences.txt'
    val_json_file = 'md_fed_data/f3set-tennis-sub/val.json'
    
    if len(sys.argv) > 1:
        val_json_file = sys.argv[1]
    
    # 读取 error_sequences.txt 中的视频名
    video_names = []
    with open(error_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and line != '------------------------' and not line.startswith('->') and '_' in line:
                # 可能是视频名
                if '/' in line or line.count('_') >= 3:
                    video_names.append(line)
    
    # 去重
    video_names = list(set(video_names))
    
    print(f"从 error_sequences.txt 提取到 {len(video_names)} 个唯一视频名")
    
    check_videos_in_val(video_names[:10], val_json_file)  # 检查前10个
