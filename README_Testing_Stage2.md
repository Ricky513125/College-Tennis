# Testing Stage 2 Model on Manually Verified Data

本指南说明如何使用训练好的 Stage 2 模型在手动验证的校园网球数据上进行测试和少样本学习。

## 目录

1. [测试 Stage 2 模型](#测试-stage-2-模型)
2. [少样本学习 (Stage 3 微调)](#少样本学习-stage-3-微调)

---

## 测试 Stage 2 模型

使用 `test_stage2_on_manual_data.py` 脚本在手动验证的数据上测试 Stage 2 模型。

### 基本用法

```bash
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/extracted/frames \
    --flow_dir /path/to/optical/flow \
    --output_dir ./test_results
```

### 参数说明

- `--checkpoint_dir`: Stage 2 模型检查点目录（必需）
  - 例如: `./MD-FED/md_fed_outputs/stage2`
  
- `--manual_annotations`: 手动标注的 JSON 文件路径（默认: `manual_annotations.json`）
  - 格式应与 `manual_annotations.json` 相同
  
- `--frame_dir`: 提取的视频帧目录（必需）
  - 应该包含按视频组织的帧文件（例如: `video_name/000000.jpg`）
  
- `--flow_dir`: 光流文件目录（可选）
  - 如果提供，将使用光流特征进行预测
  - 格式: `video_name/000000_000001.npy`
  
- `--pose_dir`: 姿态/骨架文件目录（可选）
  - Stage 2 模型不使用骨架数据，但可以保留此参数
  
- `--epoch`: 指定要加载的 epoch（可选）
  - 如果不指定，将自动使用 `loss.json` 中最佳 epoch
  - 例如: `--epoch 437`
  
- `--output_dir`: 保存测试结果的目录（可选）
  - 将保存 `test_results.json` 文件
  
- `--device`: 使用的设备（默认: `cuda`）
  - 选项: `cuda` 或 `cpu`

### 示例

```bash
# 基本测试（仅使用 RGB 帧）
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames

# 使用 RGB + 光流
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames \
    --flow_dir ~/Tennis/data/TENNIS/optical_flow \
    --output_dir ./test_results_stage2

# 指定特定 epoch
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames \
    --epoch 437
```

### 输出

脚本会输出：
- 每个视频的详细预测信息
- 评估指标（Edit Score, F1 scores）
- 如果指定了 `--output_dir`，会保存 `test_results.json`

---

## 少样本学习 (Stage 3 微调)

使用 `few_shot_learning_stage3.py` 脚本在手动验证的数据上进行少样本学习（Stage 3 微调）。

### 基本用法

```bash
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/extracted/frames \
    --flow_dir /path/to/optical/flow \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
```

### 参数说明

#### 必需参数

- `--stage2_checkpoint_dir`: Stage 2 模型检查点目录
- `--manual_annotations`: 手动标注的 JSON 文件
- `--frame_dir`: 提取的视频帧目录
- `--save_dir`: Stage 3 模型保存目录

#### 可选参数

- `--flow_dir`: 光流文件目录（推荐使用）
- `--dataset_name`: 数据集名称（默认: `ncaa-rally`）
- `--data_dir`: 保存准备数据的目录（默认: `few_shot_data`）
- `--train_ratio`: 训练集比例（默认: 0.8）
- `--num_epochs`: 训练轮数（默认: 50）
- `--batch_size`: 批次大小（默认: 4）
- `--learning_rate`: 学习率（默认: 0.0001，微调时建议使用较小值）
- `--clip_len`: 片段长度（默认: 96）
- `--crop_dim`: 裁剪尺寸（默认: 224）
- `--window`: NMS 窗口大小（默认: 5）
- `--visual_arch`: 视觉架构（默认: `rny002_tsm`）
- `--temporal_arch`: 时序架构（默认: `gru`）

### 示例

```bash
# 基本少样本学习
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames \
    --flow_dir ~/Tennis/data/TENNIS/optical_flow \
    --save_dir ./MD-FED/md_fed_outputs/stage3

# 自定义训练参数
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames \
    --flow_dir ~/Tennis/data/TENNIS/optical_flow \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 0.00005 \
    --train_ratio 0.9
```

### 训练过程

1. **数据准备**: 自动将 `manual_annotations.json` 转换为 MD-FED 格式，并分割为训练集和验证集
2. **加载 Stage 2 模型**: 从 Stage 2 检查点加载预训练模型
3. **微调**: 在手动验证的数据上进行少样本学习
4. **评估**: 在验证集上评估模型性能

### 输出

- 训练过程中的损失和评估指标
- 每个 epoch 的检查点保存在 `--save_dir`
- `loss.json`: 训练历史
- `config.json`: 训练配置

---

## 数据格式要求

### manual_annotations.json 格式

```json
[
  {
    "fps": 60.0,
    "height": 1080,
    "width": 1920,
    "num_frames": 599,
    "video": "video_name/rally_0540_0550",
    "far_name": "Player1",
    "far_hand": "RH",
    "near_name": "Player2",
    "near_hand": "LH",
    "events": [
      {
        "frame": 39,
        "label": "far_middle_serve_-_-_W_-_in"
      },
      {
        "frame": 120,
        "label": "near_deduce_return_bh_lob_DM_-_forced-err"
      }
    ]
  }
]
```

### 目录结构

```
Tennis/
├── data/
│   └── TENNIS/
│       ├── frames/              # 提取的视频帧
│       │   └── video_name/
│       │       ├── 000000.jpg
│       │       ├── 000001.jpg
│       │       └── ...
│       └── optical_flow/        # 光流文件（可选）
│           └── video_name/
│               ├── 000000_000001.npy
│               └── ...
└── manual_annotations.json      # 手动标注文件
```

---

## 常见问题

### 1. 找不到检查点文件

确保 `--checkpoint_dir` 指向正确的目录，并且包含：
- `checkpoint_XXX.pt` 文件
- `config.json` 文件
- `loss.json` 文件（可选，用于自动选择最佳 epoch）

### 2. 找不到帧文件

确保 `--frame_dir` 指向正确的目录，并且视频帧按以下结构组织：
```
frame_dir/
└── video_name/
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

其中 `video_name` 应该与 `manual_annotations.json` 中的 `video` 字段匹配。

### 3. 内存不足

如果遇到内存问题：
- 减小 `--batch_size`
- 减小 `--clip_len`
- 使用 `--device cpu`（较慢但内存占用更少）

### 4. 评估指标为 0

可能的原因：
- 标签格式不匹配
- 帧索引不匹配
- 检查 `elements.txt` 文件是否正确

---

## 下一步

完成 Stage 3 训练后，可以使用训练好的模型进行预测：

```bash
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage3 \
    --manual_annotations manual_annotations.json \
    --frame_dir ~/Tennis/data/TENNIS/frames \
    --flow_dir ~/Tennis/data/TENNIS/optical_flow
```

---

## 参考

- Stage 2 训练: `README_MD-FED_Stage2.md`
- MD-FED 设置: `README_MD-FED_setup.md`
- 数据准备: `prepare_stage2_data.py`
