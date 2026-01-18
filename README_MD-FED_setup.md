# MD-FED Stage 1 训练设置指南

本指南说明如何设置和运行 MD-FED 的 Stage 1 训练（skeleton feature pretraining）。

## 数据准备

### 1. 准备数据文件

确保你有以下文件：
- `~/Tennis/train.json` - 训练数据标注
- `~/Tennis/val.json` - 验证数据标注  
- `~/Tennis/test.json` - 测试数据标注
- `~/Tennis/data/TENNIS/skeletons/f3set-tennis/*.pkl` - Skeleton 数据文件

### 2. 合并训练数据

由于 Stage 1 可以使用所有数据，我们需要将 `train.json` 和 `val.json` 合并作为训练集，使用 `test.json` 作为验证集。

运行数据准备脚本：

```bash
python prepare_md_fed_data.py \
    --tennis_dir ~/Tennis \
    --output_dir md_fed_data \
    --dataset f3set-tennis-sub \
    --use_test_as_val
```

这个脚本会：
1. 在当前目录创建 `md_fed_data/f3set-tennis-sub/` 子文件夹
2. 合并 `train.json` + `val.json` → `md_fed_data/f3set-tennis-sub/train.json`
3. 复制 `test.json` → `md_fed_data/f3set-tennis-sub/val.json`
4. 复制 `test.json` → `md_fed_data/f3set-tennis-sub/test.json`
5. 自动从 `MD-FED/data/f3set-tennis-sub/` 复制 `elements.txt` 和 `events.txt`（如果存在）

### 3. 检查数据目录结构

数据会保存在当前目录的子文件夹中：

```
College_tennis/
├── md_fed_data/
│   └── f3set-tennis-sub/
│       ├── elements.txt          # 元素定义文件（从MD-FED/data复制）
│       ├── events.txt            # 事件定义文件（从MD-FED/data复制）
│       ├── train.json            # 训练数据（合并后的）
│       ├── val.json              # 验证数据（来自test.json）
│       └── test.json             # 测试数据
```

~/Tennis/data/TENNIS/skeletons/f3set-tennis/
├── video1.pkl
├── video2.pkl
└── ...
```

**重要**：确保 skeleton pkl 文件的名称与 JSON 文件中的 `video` 字段匹配。

例如，如果 JSON 中有：
```json
{
  "video": "20120628-M-Wimbledon-R64-Lukas_Rosol-Rafael_Nadal_101065_101157",
  ...
}
```

那么对应的 pkl 文件应该是：
```
20120628-M-Wimbledon-R64-Lukas_Rosol-Rafael_Nadal_101065_101157.pkl
```

## 训练命令

### Stage 1 训练（Skeleton Feature Pretraining）

#### 方式 1：使用辅助脚本（推荐）

使用 `run_md_fed_stage1.py` 脚本，它会自动设置符号链接并运行训练：

```bash
python run_md_fed_stage1.py \
    --data_dir md_fed_data \
    --dataset f3set-tennis-sub \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --force
```

**注意**：如果 `MD-FED/data/f3set-tennis-sub` 已经存在为目录，使用 `--force` 选项会自动备份现有目录并创建符号链接。

这个脚本会：
1. 自动在 `MD-FED/data/` 下创建符号链接指向当前目录的数据
2. 运行训练并将结果保存到 `md_fed_outputs/stage1/`

#### 方式 2：手动创建符号链接

如果你想手动控制，可以：

```bash
# 1. 创建符号链接
cd MD-FED/data
ln -s ../../md_fed_data/f3set-tennis-sub f3set-tennis-sub

# 2. 运行训练
cd ..
python3 train_MD-FED.py f3set-tennis-sub \
    --frame_dir=frames \
    --flow_dir=flows \
    --pose_dir=/home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --stage 1 \
    --visual_arch rny002_tsm \
    --skeleton_arch stgcn++ \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001 \
    -s ../md_fed_outputs/stage1
```

### 参数说明

- `f3set-tennis-sub`: 数据集名称
- `--frame_dir`: RGB 帧目录（Stage 1 可能不需要，但需要指定）
- `--flow_dir`: 光流目录（Stage 1 可能不需要，但需要指定）
- `--pose_dir`: **重要**：指向 skeleton pkl 文件目录
- `--stage 1`: Stage 1 训练（skeleton pretraining）
- `--visual_arch rny002_tsm`: 视觉架构（Stage 1 不使用，但需要指定）
- `--skeleton_arch stgcn++`: Skeleton 架构
- `--num_epochs 50`: 训练轮数
- `--batch_size 4`: 批次大小
- `--learning_rate 0.001`: 学习率
- `-s save_dir/stage1`: 保存目录

### Stage 1 的特点

- **只使用 skeleton 数据**：Stage 1 只训练 skeleton feature extractor
- **不需要 RGB 和 flow**：虽然需要指定 `--frame_dir` 和 `--flow_dir`，但 Stage 1 不会使用它们
- **训练目标**：学习从 skeleton 数据中提取特征，用于后续的 multimodal distillation

## 验证数据划分

根据你的需求：
- **训练集**：`train.json` + `val.json`（合并后）
- **验证集**：`test.json`

这样可以使用所有可用数据进行训练，同时用独立的测试集进行验证。

## 常见问题

### 1. Skeleton 文件找不到

如果遇到 skeleton 文件找不到的错误，检查：
- `--pose_dir` 路径是否正确
- pkl 文件名是否与 JSON 中的 `video` 字段匹配
- 文件权限是否正确

### 2. JSON 文件格式

确保 JSON 文件格式正确：
```json
[
  {
    "fps": 25,
    "height": 720,
    "width": 1280,
    "num_frames": 342,
    "video": "video_name",
    "events": [
      {
        "frame": 100,
        "label": "event_name"
      }
    ]
  }
]
```

### 3. 内存不足

如果遇到内存问题：
- 减小 `--batch_size`
- 减小 `--clip_len`
- 增加 `--stride`

## 下一步

Stage 1 训练完成后，可以继续：
- **Stage 2**: Multimodal Distillation（需要 RGB 和 flow 数据）
- **Stage 3**: Few-shot Fine-tuning
