# MD-FED Stage 2 训练指南

本指南说明如何运行 MD-FED 的 Stage 2 训练（Multimodal Distillation）用于 NCAA rally 数据。

## 概述

Stage 2 是无标签蒸馏阶段，使用 Stage 1 训练的 skeleton 模型作为教师模型，指导 RGB 和光流特征的学习。

## 数据要求

### 1. RGB 帧数据
- **路径**: `/mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally`
- **格式**: 每个 rally 的帧存储在 `frame_dir/video_name/rally_id/000000.jpg` 格式

### 2. 光流数据
- **路径**: `/mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally`
- **格式**: `.npy` 文件，命名格式为 `frame1_frame2.npy` (例如: `000000_000001.npy`)
- **数据形状**: `[H, W, 2]` - 每个像素的 (x, y) 光流向量

### 3. Skeleton 数据
- **路径**: 包含 `.pkl` 文件的目录
- **格式**: 每个视频一个 `.pkl` 文件，文件名应与 JSON 中的 `video` 字段匹配

### 4. 标注文件
- **文件**: `manual_annotations.json`
- **格式**: 包含所有 rally 的元数据（虽然 Stage 2 不需要标签，但需要视频元数据）

### 5. Stage 1 模型
- **路径**: `md_fed_outputs/stage1`
- **要求**: 包含训练好的 Stage 1 checkpoint 文件

## 使用步骤

### 步骤 1: 准备数据

运行数据准备脚本，将 `manual_annotations.json` 转换为 MD-FED 需要的格式：

```bash
python prepare_stage2_data.py \
    --manual_annotations ./manual_annotations.json \
    --output_dir ./md_fed_data \
    --dataset_name ncaa-rally
```

这会创建：
- `md_fed_data/ncaa-rally/train.json` - 训练数据（80%）
- `md_fed_data/ncaa-rally/val.json` - 验证数据（20%）
- `md_fed_data/ncaa-rally/elements.txt` - 元素定义（从 MD-FED 复制）
- `md_fed_data/ncaa-rally/events.txt` - 事件定义（从 MD-FED 复制）

### 步骤 2: 运行 Stage 2 训练

使用 `train_md_fed_stage2.py` 脚本进行训练：

```bash
python train_md_fed_stage2.py \
    --manual_annotations ./manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --pose_dir /mnt/ssd2/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --stage1_model_dir ./md_fed_outputs/stage1 \
    --output_dir ./md_fed_outputs/stage2 \
    --batch_size 4 \
    --num_epochs 500 \
    --learning_rate 0.001 \
    --clip_len 96 \
    --stride 2
```

### 参数说明

- `--manual_annotations`: 手动标注 JSON 文件路径
- `--frame_dir`: RGB 帧目录
- `--flow_dir`: 光流文件目录（.npy 格式）
- `--pose_dir`: Skeleton pkl 文件目录
- `--stage1_model_dir`: Stage 1 模型目录
- `--output_dir`: Stage 2 输出目录
- `--batch_size`: 批次大小（默认: 4）
- `--num_epochs`: 训练轮数（默认: 50）
- `--learning_rate`: 学习率（默认: 0.001）
- `--clip_len`: 每个 clip 的帧数（默认: 96）
- `--stride`: 帧采样步长（默认: 2）
- `--visual_arch`: 视觉架构（默认: rny002_tsm）
- `--skeleton_arch`: Skeleton 架构（默认: stgcn++）

### 仅准备数据（不训练）

如果只想准备数据而不立即训练：

```bash
python train_md_fed_stage2.py \
    --manual_annotations ./manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --pose_dir /path/to/skeleton/pkl/files \
    --prepare_data_only
```

## 技术细节

### 光流文件适配

脚本会自动处理 `.npy` 格式的光流文件：
- MD-FED 原本期望 JPG 格式的光流
- 我们的光流文件是 `.npy` 格式，命名格式为 `frame1_frame2.npy`
- `flow_adapter.py` 模块会动态 patch MD-FED 的 `FrameReader` 类来支持 `.npy` 文件

### 数据流程

1. **数据准备**: `prepare_stage2_data.py` 从 `manual_annotations.json` 创建训练/验证 JSON
2. **符号链接**: 在 `MD-FED/data/` 下创建符号链接指向我们的数据目录
3. **Flow 适配**: `flow_adapter.py` 修改 FrameReader 以支持 `.npy` 格式
4. **训练**: 调用 MD-FED 的训练脚本进行 Stage 2 训练

### Stage 2 训练特点

- **无标签蒸馏**: Stage 2 不需要事件标签，只使用 RGB、光流和 skeleton 特征
- **损失函数**: L2 损失，使 RGB 和光流特征与 skeleton 特征对齐
  - `rgb2sk_loss = MSE(rgb_feat, sk_feat)`
  - `flow2sk_loss = MSE(flow_feat, sk_feat)`
- **教师模型**: 使用 Stage 1 训练的 skeleton 模型作为教师

## 输出

训练完成后，输出目录包含：

```
md_fed_outputs/stage2/
├── checkpoint_000.pt
├── checkpoint_001.pt
├── ...
├── checkpoint_049.pt
├── best.pt
├── config.txt
└── history.json
```

- `checkpoint_XXX.pt`: 每个 epoch 的 checkpoint
- `best.pt`: 验证损失最低的模型
- `config.txt`: 训练配置
- `history.json`: 训练历史

## 常见问题

### 1. 光流文件找不到

确保：
- 光流文件命名格式正确：`frame1_frame2.npy`
- 文件路径与 JSON 中的 `video` 字段匹配
- 目录结构正确：`flow_dir/video_name/000000_000001.npy`

### 2. Skeleton 文件找不到

确保：
- `.pkl` 文件名与 JSON 中的 `video` 字段匹配
- 文件路径正确
- 文件格式正确（包含 `keypoint` 字段）

### 3. Stage 1 模型加载失败

检查：
- `--stage1_model_dir` 路径是否正确
- 目录中是否包含 `checkpoint_XXX.pt` 文件
- 是否有 `history.json` 文件用于确定最佳 epoch

### 4. 内存不足

如果遇到内存问题：
- 减小 `--batch_size`
- 减小 `--clip_len`
- 增加 `--stride`

## 下一步

Stage 2 训练完成后，可以继续：
- **Stage 3**: Few-shot Fine-tuning（需要少量标注数据）
