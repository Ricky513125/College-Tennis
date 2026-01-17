# Skeleton Annotation using HRNet

使用HRNet模型为视频生成skeleton（姿态）标注。

## 模型位置

模型文件位于：
```
./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth
```

## 使用方法

### 基本用法

```bash
python generate_skeleton_annotations.py \
    /path/to/video.mp4 \
    --model_path ./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth \
    --output ./video_skeleton.json
```

### 完整参数

```bash
python generate_skeleton_annotations.py \
    /path/to/video.mp4 \
    --model_path ./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth \
    --config ./deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
    --output ./video_skeleton.json \
    --inference_fps 10 \
    --detection_threshold 0.9
```

## 参数说明

- `video_path`: **必需**。输入视频文件路径
- `--model_path`: HRNet模型路径（默认：`./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth`）
- `--config`: 配置文件路径（可选，会自动检测）
- `--output`: 输出JSON文件路径（默认：`<video_name>_skeleton.json`）
- `--inference_fps`: 推理FPS（默认：处理所有帧）
- `--detection_threshold`: 人体检测阈值（默认：0.9）

## 输出格式

生成的JSON文件包含以下结构：

```json
{
  "video_path": "/path/to/video.mp4",
  "fps": 29.97,
  "total_frames": 1000,
  "processed_frames": 1000,
  "width": 1920,
  "height": 1080,
  "keypoint_names": [
    "nose", "left_eye", "right_eye", ...
  ],
  "annotations": [
    {
      "frame": 0,
      "persons": [
        {
          "person_id": 0,
          "bbox": [[x1, y1], [x2, y2]],
          "keypoints": {
            "nose": {"x": 100.5, "y": 200.3, "visible": 1.0},
            "left_eye": {"x": 95.2, "y": 195.1, "visible": 1.0},
            ...
          }
        }
      ]
    },
    ...
  ]
}
```

## 关键点说明

模型检测17个COCO格式的关键点：
1. nose (鼻子)
2. left_eye, right_eye (左右眼)
3. left_ear, right_ear (左右耳)
4. left_shoulder, right_shoulder (左右肩)
5. left_elbow, right_elbow (左右肘)
6. left_wrist, right_wrist (左右腕)
7. left_hip, right_hip (左右髋)
8. left_knee, right_knee (左右膝)
9. left_ankle, right_ankle (左右踝)

## 处理NCAA视频

### 处理单个视频

```bash
python generate_skeleton_annotations.py \
    ../ncaa_videos/RUokidaZR30.mp4 \
    --model_path ./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth \
    --output ./RUokidaZR30_skeleton.json
```

### 批量处理

可以创建一个简单的bash脚本来处理多个视频：

```bash
#!/bin/bash
for video in ../ncaa_videos/*.mp4; do
    video_name=$(basename "$video" .mp4)
    python generate_skeleton_annotations.py \
        "$video" \
        --model_path ./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth \
        --output "./${video_name}_skeleton.json"
done
```

## 性能说明

- **处理速度**: 在GPU上，每帧大约需要0.13秒（包括人体检测和姿态估计）
- **内存使用**: 需要足够的GPU内存来加载模型
- **CPU模式**: 如果没有GPU，会自动使用CPU，但速度会慢很多

## 注意事项

1. **模型文件**: 确保模型文件路径正确
2. **GPU**: 建议使用GPU加速，CPU模式会很慢
3. **内存**: 处理长视频时可能需要较多内存
4. **多人检测**: 脚本会自动检测并标注视频中的所有人物
5. **输出位置**: 所有输出文件默认保存在当前目录（College_tennis）下

## 示例

```bash
# 处理一个视频，每10帧处理一次（加速）
python generate_skeleton_annotations.py \
    ../ncaa_videos/6VSmpCSgY7M.mp4 \
    --model_path ./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth \
    --output ./6VSmpCSgY7M_skeleton.json \
    --inference_fps 10
```
