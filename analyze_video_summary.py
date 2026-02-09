#!/usr/bin/env python3
"""
统计 manual_annotations.json 中各个视频的总结信息
"""

import json
from collections import defaultdict

def main():
    # 读取数据
    with open('manual_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按视频ID分组统计
    video_stats = defaultdict(lambda: {
        'rallies': 0,
        'total_frames': 0,
        'fps_sum': 0,
        'duration': 0,
        'events': 0
    })
    
    for item in data:
        video_name = item.get('video', '')
        # 提取视频ID（去掉rally部分）
        if '/' in video_name:
            video_id = video_name.split('/')[0]
        else:
            video_id = video_name
        
        num_frames = item.get('num_frames', 0)
        fps = item.get('fps', 0)
        num_events = len(item.get('events', []))
        duration = num_frames / fps if fps > 0 else 0
        
        video_stats[video_id]['rallies'] += 1
        video_stats[video_id]['total_frames'] += num_frames
        video_stats[video_id]['fps_sum'] += fps
        video_stats[video_id]['duration'] += duration
        video_stats[video_id]['events'] += num_events
    
    # 计算每个视频的平均FPS
    for video_id in video_stats:
        rallies = video_stats[video_id]['rallies']
        if rallies > 0:
            video_stats[video_id]['avg_fps'] = video_stats[video_id]['fps_sum'] / rallies
    
    # 视频ID友好名称映射
    video_id_names = {
        '6VSmpCSgY7M': '6VSmpCSgY7M',
        'Avendano__UL__Vs__Penzlin__LSU_': 'Avendano vs Penzlin',
        'dwPey52i1LE': 'dwPey52i1LE',
        'Hoole_SC_vs_Dong_LSU': 'Hoole vs Dong',
        'IohTeru65U4': 'IohTeru65U4',
        'Lc9MSf6vHxU': 'Lc9MSf6vHxU',
        'RUokidaZR30': 'RUokidaZR30'
    }
    
    # 输出到文件
    output_file = 'video_summary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Video Summary Statistics\n")
        f.write("="*100 + "\n\n")
        
        # 表头
        f.write(f"{'Video ID':<30} {'Rallies':>10} {'Total Frames':>15} {'FPS':>10} {'Duration (s)':>15} {'Events':>10}\n")
        f.write("-"*100 + "\n")
        
        # 排序并输出每个视频
        total_rallies = 0
        total_frames = 0
        total_duration = 0
        total_events = 0
        
        # 按视频ID排序
        sorted_videos = sorted(video_stats.items(), key=lambda x: x[0])
        
        for video_id, stats in sorted_videos:
            # 使用友好名称（如果有的话）
            display_name = video_id_names.get(video_id, video_id)
            
            rallies = stats['rallies']
            frames = stats['total_frames']
            avg_fps = stats['avg_fps']
            duration = stats['duration']
            events = stats['events']
            
            f.write(f"{display_name:<30} {rallies:>10} {frames:>15,} {avg_fps:>10.2f} {duration:>15.2f} {events:>10}\n")
            
            total_rallies += rallies
            total_frames += frames
            total_duration += duration
            total_events += events
        
        # 输出总计
        f.write("-"*100 + "\n")
        f.write(f"{'Total':<30} {total_rallies:>10} {total_frames:>15,} {'–':>10} {total_duration:>15.2f} {total_events:>10}\n")
        f.write("="*100 + "\n\n")
        
        # 额外的统计信息
        f.write("\nAdditional Statistics:\n")
        f.write("-"*100 + "\n")
        f.write(f"Average rallies per video: {total_rallies / len(video_stats):.2f}\n")
        f.write(f"Average frames per video: {total_frames / len(video_stats):,.2f}\n")
        f.write(f"Average duration per video: {total_duration / len(video_stats):.2f}s\n")
        f.write(f"Average events per video: {total_events / len(video_stats):.2f}\n")
        f.write(f"Average frames per rally: {total_frames / total_rallies:.2f}\n")
        f.write(f"Average events per rally: {total_events / total_rallies:.2f}\n")
        f.write(f"Average duration per rally: {total_duration / total_rallies:.2f}s\n")
        
        # 按视频详细信息
        f.write("\n" + "="*100 + "\n")
        f.write("Detailed Video Information\n")
        f.write("="*100 + "\n\n")
        
        for video_id, stats in sorted_videos:
            display_name = video_id_names.get(video_id, video_id)
            f.write(f"\n{display_name} ({video_id}):\n")
            f.write(f"  Rallies: {stats['rallies']}\n")
            f.write(f"  Total Frames: {stats['total_frames']:,}\n")
            f.write(f"  Average FPS: {stats['avg_fps']:.2f}\n")
            f.write(f"  Total Duration: {stats['duration']:.2f}s ({stats['duration']/60:.2f} min)\n")
            f.write(f"  Total Events: {stats['events']}\n")
            f.write(f"  Average frames per rally: {stats['total_frames'] / stats['rallies']:.2f}\n")
            f.write(f"  Average events per rally: {stats['events'] / stats['rallies']:.2f}\n")
            f.write(f"  Average duration per rally: {stats['duration'] / stats['rallies']:.2f}s\n")
    
    print(f"Video summary saved to: {output_file}")
    
    # 同时输出到控制台
    print("\n" + "="*100)
    print("Video Summary Statistics")
    print("="*100 + "\n")
    
    print(f"{'Video ID':<30} {'Rallies':>10} {'Total Frames':>15} {'FPS':>10} {'Duration (s)':>15} {'Events':>10}")
    print("-"*100)
    
    for video_id, stats in sorted_videos:
        display_name = video_id_names.get(video_id, video_id)
        rallies = stats['rallies']
        frames = stats['total_frames']
        avg_fps = stats['avg_fps']
        duration = stats['duration']
        events = stats['events']
        
        print(f"{display_name:<30} {rallies:>10} {frames:>15,} {avg_fps:>10.2f} {duration:>15.2f} {events:>10}")
    
    print("-"*100)
    print(f"{'Total':<30} {total_rallies:>10} {total_frames:>15,} {'–':>10} {total_duration:>15.2f} {total_events:>10}")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
