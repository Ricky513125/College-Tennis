#!/usr/bin/env python3
"""
统计 manual_annotations.json 的数据
"""

import json
from collections import defaultdict

# 定义映射关系
MAPPING = {
    # sc1: near/far
    'near': 'near',
    'far': 'far',
    
    # sc2: deuce/ad/middle
    'deuce': 'deuce',
    'ad': 'ad',
    'middle': 'middle',
    'deduce': 'deuce',  # 可能是拼写错误
    
    # sc3: forehand/backhand
    'fh': 'forehand',
    'bh': 'backhand',
    
    # sc4: serve/return/stroke
    'serve': 'serve',
    'return': 'return',
    'stroke': 'stroke',
    
    # sc5: directions
    'T': 'T',
    'W': 'Wide',
    'DM': 'down the middle',
    'DL': 'down the line',
    'CC': 'cross-court',
    'II': 'inside-in',
    'IO': 'inside-out',
    # Body 在标签中没有明确标记
    
    # sc6: techniques
    'gs': 'ground stroke',
    'slice': 'slice',
    'volley': 'volley',
    'lob': 'lob',
    'drop': 'drop',
    'smash': 'smash',
    
    # sc7: approach
    'approach': 'approach',
    
    # sc8: results
    'in': 'in-bound',
    'winner': 'winner',
    'forced-err': 'forced error',
    'unforced-err': 'unforced error',
}

def parse_label(label_str):
    """
    解析标签字符串，提取各个子类别信息
    
    标签格式示例：
    - "far_middle_serve_-_-_W_-_in"
    - "near_deduce_return_bh_lob_DM_-_forced-err"
    - "far_deuce_stroke_fh_gs_CC_-_in"
    """
    parts = label_str.split('_')
    
    info = {
        'sc1': None,  # near/far
        'sc2': None,  # deuce/ad/middle
        'sc3': None,  # forehand/backhand
        'sc4': None,  # serve/return/stroke
        'sc5': None,  # direction
        'sc6': None,  # technique
        'sc7': 'non-approach',  # approach
        'sc8': None,  # result
    }
    
    # sc1: near/far (第一个位置)
    if parts[0] in ['near', 'far']:
        info['sc1'] = parts[0]
    
    # sc2: deuce/ad/middle (第二个位置)
    if len(parts) > 1:
        if parts[1] in ['deuce', 'ad', 'middle', 'deduce']:
            info['sc2'] = 'deuce' if parts[1] == 'deduce' else parts[1]
    
    # sc4: serve/return/stroke (第三个位置)
    if len(parts) > 2:
        if parts[2] in ['serve', 'return', 'stroke']:
            info['sc4'] = parts[2]
    
    # 遍历所有部分
    for i, part in enumerate(parts):
        # sc3: fh/bh
        if part in ['fh', 'bh']:
            info['sc3'] = 'forehand' if part == 'fh' else 'backhand'
        
        # sc6: technique
        if part in ['gs', 'slice', 'volley', 'lob', 'drop', 'smash']:
            info['sc6'] = 'ground stroke' if part == 'gs' else part
        
        # sc5: direction
        if part in ['T', 'W', 'DM', 'DL', 'CC', 'II', 'IO']:
            if part == 'W':
                info['sc5'] = 'Wide'
            elif part == 'DM':
                info['sc5'] = 'down the middle'
            elif part == 'DL':
                info['sc5'] = 'down the line'
            elif part == 'CC':
                info['sc5'] = 'cross-court'
            elif part == 'II':
                info['sc5'] = 'inside-in'
            elif part == 'IO':
                info['sc5'] = 'inside-out'
            else:
                info['sc5'] = part
        
        # sc7: approach
        if part == 'approach':
            info['sc7'] = 'approach'
        
        # sc8: result
        if part == 'in':
            info['sc8'] = 'in-bound'
        elif part == 'winner':
            info['sc8'] = 'winner'
        elif part == 'forced-err':
            info['sc8'] = 'forced error'
        elif part == 'unforced-err':
            info['sc8'] = 'unforced error'
    
    return info


def main():
    # 读取数据
    with open('manual_annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打开输出文件
    output_file = 'manual_annotations_statistics.txt'
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"{'='*80}\n")
        out.write(f"Manual Annotations Statistics\n")
        out.write(f"{'='*80}\n\n")
        
        # 统计视频数量和片段信息
        out.write(f"Total videos: {len(data)}\n")
        
        # 统计每个视频的长度
        total_frames = 0
        total_events = 0
        video_lengths = []
        
        for item in data:
            num_frames = item.get('num_frames', 0)
            num_events = len(item.get('events', []))
            video_lengths.append(num_frames)
            total_frames += num_frames
            total_events += num_events
        
        out.write(f"Total events: {total_events}\n")
        out.write(f"Total frames: {total_frames:,}\n")
        out.write(f"Average video length: {total_frames / len(data):.1f} frames\n")
        out.write(f"Min video length: {min(video_lengths)} frames\n")
        out.write(f"Max video length: {max(video_lengths)} frames\n")
        out.write(f"Average events per video: {total_events / len(data):.2f}\n\n")
    
        # 统计各子类别
        sc_counts = {
            'sc1': defaultdict(int),
            'sc2': defaultdict(int),
            'sc3': defaultdict(int),
            'sc4': defaultdict(int),
            'sc5': defaultdict(int),
            'sc6': defaultdict(int),
            'sc7': defaultdict(int),
            'sc8': defaultdict(int),
        }
        
        # 遍历所有事件
        for item in data:
            for event in item.get('events', []):
                label = event.get('label', '')
                info = parse_label(label)
                
                # 统计每个子类别
                for sc_key, sc_value in info.items():
                    if sc_value:
                        sc_counts[sc_key][sc_value] += 1
        
        # 写入统计结果
        out.write(f"{'='*80}\n")
        out.write(f"Sub-Class Element Statistics\n")
        out.write(f"{'='*80}\n\n")
        
        sc_names = {
            'sc1': 'SC1: Player Position (near/far)',
            'sc2': 'SC2: Court Position (deuce/ad/middle)',
            'sc3': 'SC3: Hand (forehand/backhand)',
            'sc4': 'SC4: Action Type (serve/return/stroke)',
            'sc5': 'SC5: Direction',
            'sc6': 'SC6: Technique',
            'sc7': 'SC7: Approach',
            'sc8': 'SC8: Result',
        }
        
        # 定义显示顺序
        display_order = {
            'sc1': ['near', 'far'],
            'sc2': ['deuce', 'ad', 'middle'],
            'sc3': ['forehand', 'backhand'],
            'sc4': ['serve', 'return', 'stroke'],
            'sc5': ['T', 'Wide', 'cross-court', 'down the line', 'down the middle', 
                    'inside-in', 'inside-out'],
            'sc6': ['ground stroke', 'slice', 'volley', 'lob', 'drop', 'smash'],
            'sc7': ['approach', 'non-approach'],
            'sc8': ['in-bound', 'winner', 'forced error', 'unforced error'],
        }
        
        for sc_key in ['sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6', 'sc7', 'sc8']:
            out.write(f"\n{sc_names[sc_key]}\n")
            out.write(f"{'-'*80}\n")
            out.write(f"{'Element':<25} {'Count':>10} {'Proportion':>15}\n")
            out.write(f"{'-'*80}\n")
            
            total = sum(sc_counts[sc_key].values())
            
            # 按照定义的顺序显示
            for element in display_order.get(sc_key, []):
                count = sc_counts[sc_key].get(element, 0)
                if total > 0:
                    proportion = count / total * 100
                    out.write(f"{element:<25} {count:>10,} {proportion:>14.1f}%\n")
            
            # 显示其他未定义的元素
            for element, count in sorted(sc_counts[sc_key].items()):
                if element not in display_order.get(sc_key, []):
                    if total > 0:
                        proportion = count / total * 100
                        out.write(f"{element:<25} {count:>10,} {proportion:>14.1f}%\n")
            
            out.write(f"{'-'*80}\n")
            out.write(f"{'Total':<25} {total:>10,} {100.0:>14.1f}%\n")
    
    # 保存详细的视频信息到另一个文件
    print(f"Statistics saved to: {output_file}")
    print(f"Generating detailed video information...")
    
    with open('manual_annotations_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Manual Annotations - Detailed Video Information\n")
        f.write(f"{'='*80}\n\n")
        
        for i, item in enumerate(data, 1):
            video_name = item.get('video', 'Unknown')
            num_frames = item.get('num_frames', 0)
            num_events = len(item.get('events', []))
            fps = item.get('fps', 0)
            duration = num_frames / fps if fps > 0 else 0
            
            f.write(f"{i}. {video_name}\n")
            f.write(f"   Frames: {num_frames}, Events: {num_events}, ")
            f.write(f"FPS: {fps:.2f}, Duration: {duration:.2f}s\n")
            
            for j, event in enumerate(item.get('events', []), 1):
                frame = event.get('frame', 0)
                label = event.get('label', '')
                time = frame / fps if fps > 0 else 0
                f.write(f"   [{j}] Frame {frame} ({time:.2f}s): {label}\n")
            
            f.write("\n")
    
    print(f"Detailed video information saved to: manual_annotations_summary.txt")
    print(f"All done!")


if __name__ == '__main__':
    main()
