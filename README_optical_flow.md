# Optical Flow Generation using RAFT

使用RAFT模型为视频生成光流（optical flow）文件。

## 模型位置

模型文件位于：
```
./RAFT/models/raft-small.pth
```

## 使用方法

### 处理手动指定的Rally（推荐）

如果你已经使用 `process_manual_rallies.py` 提取了rally的帧，可以使用批量处理脚本为所有rally生成optical flow：

```bash
# 1. 首先提取rally帧（如果还没做）
python process_manual_rallies.py \
    --config example_rally_config.json \
    --output_dir ./ncaa_annotations_rally \
    --frame_dir ./ncaa_frames_rally

# 2. 为所有rally生成optical flow
python batch_generate_optical_flow.py \
    --metadata ./ncaa_annotations_rally/rallies_metadata.json \
    --frame_dir ./ncaa_frames_rally \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow_rally
```

这会为每个rally生成独立的optical flow文件，保存在 `output_dir` 中。

### 批量处理完整视频

```bash
bash batch_flow_6_videos.sh
```

或者使用Python脚本：

```bash
python batch_generate_optical_flow.py \
    --metadata ./ncaa_annotations/ncaa_videos_metadata.json \
    --frame_dir ./ncaa_frames \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow \
    --skip_frames 1
```

### 处理单个视频

```bash
python generate_optical_flow.py \
    ./ncaa_frames/6VSmpCSgY7M \
    --video_id 6VSmpCSgY7M \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow \
    --use_frames


python generate_optical_flow.py \
    ./ncaa_frames/Hoole_SC_vs_Dong_LSU \
    --video_id Hoole_SC_vs_Dong_LSU.mp4 \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow \
    --use_frames

```

## 参数说明

- `input_path`: 帧目录路径
- `--video_id`: 视频ID
- `--model_path`: RAFT模型路径（默认：`./RAFT/models/raft-small.pth`）
- `--output_dir`: 输出目录（默认：`./ncaa_optical_flow`）
- `--skip_frames`: 处理间隔（默认：1 = 所有帧）
- `--iters`: RAFT迭代次数（默认：20）
- `--small`: 使用small模型（默认：True）

## 输出格式

每个视频会在输出目录下创建一个子目录：

```
ncaa_optical_flow/
├── 6VSmpCSgY7M/
│   ├── 000000_000001.npy  # 从帧0到帧1的光流
│   ├── 000001_000002.npy  # 从帧1到帧2的光流
│   ├── ...
│   └── flow_metadata.json # 元数据文件
├── Avendano__UL__Vs__Penzlin__LSU_/
│   └── ...
└── ...
```

### 光流文件格式

- **文件格式**: `.npy` (NumPy数组)
- **数据形状**: `[H, W, 2]` - 每个像素的 (x, y) 光流向量
- **命名**: `{frame1:06d}_{frame2:06d}.npy` - 表示从frame1到frame2的光流

### 元数据文件

每个视频目录包含 `flow_metadata.json`：

```json
{
  "video_id": "6VSmpCSgY7M",
  "frame_dir": "./ncaa_frames/6VSmpCSgY7M",
  "fps": 29.97,
  "total_frames": 408203,
  "processed_pairs": 408202,
  "width": 398,
  "height": 224,
  "flow_files": [
    {
      "frame1": 0,
      "frame2": 1,
      "flow_file": "000000_000001.npy",
      "flow_shape": [224, 398, 2]
    },
    ...
  ]
}
```

## 优势

- **复用已提取的帧**: 直接使用F3-set提取的帧图片，无需重新处理视频
- **高效**: 跳过视频解码步骤
- **批量处理**: 一次处理多个视频
- **灵活**: 可设置跳过帧数来加速处理

## 性能说明

- **处理速度**: 在GPU上，每对帧大约需要0.1-0.2秒
- **内存使用**: 需要足够的GPU内存
- **CPU模式**: 如果没有GPU，会自动使用CPU，但速度会慢很多

## 注意事项

1. **模型文件**: 确保模型文件路径正确
2. **GPU**: 建议使用GPU加速，CPU模式会很慢
3. **内存**: 处理长视频时可能需要较多内存
4. **输出位置**: 所有输出文件保存在 `./ncaa_optical_flow/` 目录下

## 示例

### 处理Rally（推荐工作流）

```bash
# 步骤1: 提取rally帧
python process_manual_rallies.py \
    --config example_rally_config.json \
    --output_dir ./ncaa_annotations_rally \
    --frame_dir ./ncaa_frames_rally

# 步骤2: 为所有rally生成optical flow
python batch_generate_optical_flow.py \
    --metadata ./ncaa_annotations_rally/rallies_metadata.json \
    --frame_dir ./ncaa_frames_rally \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow_rally \
    --skip_frames 1

# 或者只处理特定的rally（通过video_id）
python batch_generate_optical_flow.py \
    --metadata ./ncaa_annotations_rally/rallies_metadata.json \
    --frame_dir ./ncaa_frames_rally \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow_rally \
    --video_list "6VSmpCSgY7M/rally_0540_0550" "6VSmpCSgY7M/rally_0605_0610"
```

### 处理完整视频

```bash
# 批量处理所有视频
python batch_generate_optical_flow.py \
    --metadata ./ncaa_annotations/ncaa_videos_metadata.json \
    --frame_dir ./ncaa_frames \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow

# 处理单个视频，每2帧处理一次（加速）
python generate_optical_flow.py \
    ./ncaa_frames/6VSmpCSgY7M \
    --video_id 6VSmpCSgY7M \
    --model_path ./RAFT/models/raft-small.pth \
    --output_dir ./ncaa_optical_flow \
    --skip_frames 2 \
    --use_frames
```
