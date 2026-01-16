# 快速开始指南

## 你的模型位置
模型位于：`F3Set/f3set-model/f3ed/`

包含文件：
- `checkpoint_049.pt` - 模型检查点
- `config.json` - 模型配置
- `loss.json` - 训练损失历史

## 一键运行（推荐）

### Linux/Mac:
```bash
python run_full_pipeline.py \
    --video_dir ../ncaa_videos \
    --model_dir ./F3Set/f3set-model/f3ed \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames \
    --dataset f3set-tennis \
    --use_f3ed
```

### Windows:
```cmd
python run_full_pipeline.py ^
    --video_dir ..\ncaa_videos ^
    --model_dir .\F3Set\f3set-model\f3ed ^
    --output_dir .\ncaa_annotations ^
    --frame_dir .\ncaa_frames ^
    --dataset f3set-tennis ^
    --use_f3ed
```

## 分步运行

### 步骤1: 提取帧
```bash
python process_ncaa_videos.py \
    --video_dir ../ncaa_videos \
    --output_dir ./ncaa_annotations \
    --frame_dir ./ncaa_frames
```

### 步骤2: 生成标注
```bash
python generate_annotations.py \
    --metadata ./ncaa_annotations/ncaa_videos_metadata.json \
    --frame_dir ./ncaa_frames \
    --model_dir ./F3Set/f3set-model/f3ed \
    --output_dir ./ncaa_annotations \
    --dataset f3set-tennis \
    --use_f3ed
```

## 输出文件

运行完成后，在 `./ncaa_annotations/` 目录下会生成：
- `ncaa_videos_metadata.json` - 视频元数据
- `annotations.json` - **最终的标注文件**（包含所有检测到的事件）

## 注意事项

1. **视频路径**: 确保 `../ncaa_videos` 目录存在且包含视频文件
2. **GPU**: 如果有GPU，脚本会自动使用；否则使用CPU（较慢）
3. **内存**: 处理大视频可能需要较多内存
4. **时间**: 帧提取和推理都需要一些时间，请耐心等待

## 如果遇到问题

1. **找不到模型**: 检查 `./F3Set/f3set-model/f3ed/` 目录是否存在
2. **找不到视频**: 检查 `../ncaa_videos` 目录是否存在
3. **导入错误**: 确保在 `College_tennis` 目录下运行脚本
4. **CUDA错误**: 如果没有GPU，脚本会自动使用CPU
