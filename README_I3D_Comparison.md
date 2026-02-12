# I3D 对比实验说明

## 📋 概述

本文档介绍如何使用 I3D（Inflated 3D ConvNet）模型进行对比实验，与 MD-FED Stage 3 和 VTN 模型进行性能比较。

## 🏗️ I3D 模型架构

I3D 是一个经典的视频理解模型，其核心特点：

1. **3D 卷积**：直接在时空维度上进行卷积，同时捕捉空间和时间信息
2. **Inception 架构**：使用 Inception-v1 架构的 3D 版本
3. **双流架构**：原始 I3D 支持 RGB 和 Optical Flow 双流，本实验只使用 RGB 流
4. **预训练**：可使用 Kinetics 数据集预训练的权重

## 🚀 快速开始

### 基础训练命令

```bash
python train_i3d_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./i3d_outputs \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500
```

### 使用预训练权重

如果你有 Kinetics 预训练的 I3D 权重：

```bash
python train_i3d_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./i3d_outputs \
    --pretrained_path /path/to/rgb_imagenet.pt \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500
```

### 从头训练（不使用预训练）

```bash
python train_i3d_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./i3d_outputs \
    --no_pretrained \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500
```

## 📊 参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manual_annotations` | 必需 | 标注文件路径 |
| `--frame_dir` | 必需 | 帧目录路径 |
| `--save_dir` | `./i3d_outputs` | 模型保存目录 |
| `--crop_dim` | `224` | 图像裁剪尺寸 |
| `--clip_len` | `96` | 每个 clip 的帧数 |
| `--stride` | `2` | 帧采样步长 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dropout` | `0.5` | Dropout 比率 |
| `--no_pretrained` | `False` | 不使用预训练权重 |
| `--pretrained_path` | `None` | 预训练权重路径 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | `4` | 批次大小 |
| `--num_epochs` | `500` | 训练轮数 |
| `--learning_rate` | `1e-4` | 初始学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_epochs` | `10` | Warmup 轮数 |
| `--dataset_len` | `1000` | 每个 epoch 的 clip 数 |

## 💡 关键设计

### 1. 与 MD-FED 的公平对比

- ✅ **相同的数据分割**：80% 训练，20% 验证
- ✅ **相同的数据增强**：训练时随机裁剪 224×224，验证时居中裁剪 224×224
- ✅ **相同的评估指标**：F1-score, Edit Score
- ✅ **相同的输入**：RGB 帧，clip_len=96

### 2. 模型架构

```
输入: [batch, 96, 3, 224, 224]
  ↓
I3D Backbone (Mixed_4f)
  ↓
[batch, 832, 48, 7, 7]
  ↓
Spatial Pooling
  ↓
[batch, 832, 48]
  ↓
GRU Temporal Head
  ↓
[batch, 48, 368]
  ↓
Temporal Upsampling
  ↓
[batch, 96, 368]
  ↓
Classification Heads
  ↓
Coarse: [batch, 96, 2]
Fine: [batch, 96, num_classes]
```

### 3. 特征提取层选择

使用 `Mixed_4f` 作为特征提取层，原因：
- ✅ 避免过拟合（相比完整的 I3D）
- ✅ 计算效率高
- ✅ 特征维度适中（832 维）

## 📈 训练监控

训练时会显示：

```
Epoch 1/500 [Train]: 100%|████████| 19/19 [01:23<00:00, loss: 2.3456]
Epoch 1/500 [Val]: 100%|████████| 5/5 [00:15<00:00, loss: 2.1234]
[Epoch 1/500] Train: 2.3456 | Val: 2.1234 | LR: 1.00e-04
  ✅ New best val loss: 2.1234 → Saving checkpoint...
```

## 📁 输出文件

训练完成后，`save_dir` 中会包含：

```
i3d_outputs/
├── train_annotations.json      # 训练集标注
├── val_annotations.json        # 验证集标注
├── elements.txt                # 类别定义
├── best_model.pt              # 最佳模型 ⭐
├── checkpoint_000.pt          # Epoch 0 模型
├── checkpoint_001.pt          # Epoch 1 模型
├── ...
└── loss.json                  # 训练损失记录
```

## 🔬 下载预训练权重（可选）

如果你想使用 Kinetics 预训练的 I3D 权重：

```bash
# 从常见来源下载
# 例如：https://github.com/piergiaj/pytorch-i3d

# 或使用 PyTorch Hub (如果可用)
python -c "
import torch
model = torch.hub.load('pytorch/vision', 'i3d_r50', pretrained=True)
torch.save(model.state_dict(), 'rgb_imagenet.pt')
"
```

**注意**：预训练权重不是必需的，从头训练也能获得不错的效果。

## 🎯 与 VTN 的对比

| 特性 | I3D | VTN |
|------|-----|-----|
| **架构** | 3D CNN | ViT + Transformer |
| **参数量** | ~12M | ~22M (small) |
| **计算复杂度** | 高 | 中等 |
| **预训练** | Kinetics | ImageNet |
| **时空建模** | 同时（3D卷积） | 分离（空间→时间） |
| **显存占用** | 较高 | 中等 |

## 💻 显存和性能建议

### 显存受限

如果显存不足，可以：

1. **减小 batch_size**：
   ```bash
   --batch_size 2
   ```

2. **减小 clip_len**：
   ```bash
   --clip_len 64
   ```

3. **使用较浅的特征层**：
   ```bash
   # 修改代码中的 final_endpoint='Mixed_3c'  # 更浅的层
   ```

### 优化训练速度

1. **增加 workers**：
   ```bash
   --num_workers 8
   ```

2. **减小 dataset_len**：
   ```bash
   --dataset_len 500
   ```

## 🔍 性能评估

训练完成后，使用与 MD-FED Stage 3 相同的评估脚本：

```bash
python evaluate_per_video.py \
    --checkpoint ./i3d_outputs/best_model.pt \
    --model_type i3d \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally
```

## 🎓 参考文献

```bibtex
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
```

## 📝 常见问题

### Q1: I3D 的 batch_size 为什么比 VTN 小？

A: I3D 使用 3D 卷积，显存占用比 VTN 高。建议从 batch_size=2 开始，根据显存情况调整。

### Q2: 需要光流（Optical Flow）吗？

A: 本实验只使用 RGB 流，不需要光流。原始 I3D 论文使用双流（RGB + Flow），但单 RGB 流已经足够。

### Q3: 预训练权重重要吗？

A: 对于小数据集，预训练权重有帮助但不是必需的。你可以先尝试不使用预训练，如果效果不好再考虑预训练。

### Q4: I3D 训练比 VTN 慢吗？

A: 是的，I3D 的 3D 卷积比 VTN 的 2D 卷积 + Transformer 慢，大约慢 1.5-2 倍。

## 🚨 故障排除

### OOM (Out of Memory)

```bash
# 解决方案 1: 减小 batch_size
--batch_size 2

# 解决方案 2: 减小 clip_len
--clip_len 64

# 解决方案 3: 减小图像尺寸
--crop_dim 112
```

### 训练不收敛

```bash
# 增加 warmup
--warmup_epochs 20

# 降低学习率
--learning_rate 5e-5

# 增加正则化
--dropout 0.7
--weight_decay 1e-3
```

## 📞 支持

如有问题，请参考：
- `train_i3d_comparison.py` 源码
- `README_VTN_Comparison.md`（类似的对比实验）
- `README_Fair_Comparison.md`（公平对比策略）
