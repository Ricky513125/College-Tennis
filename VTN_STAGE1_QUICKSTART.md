# VTN Stage 1 快速开始

## 🎯 目标

在 F3Set 数据集上预训练 VTN，为 Stage 3 微调提供更好的初始化。

---

## 📋 前置条件

### 1. 准备 F3Set 帧数据

确保已经提取了 F3Set 的视频帧：

```bash
# 检查 F3Set 数据结构
ls F3Set/data/f3set-tennis/
# 应该包含: train.json, val.json, elements.txt

# 检查帧目录 (需要自己提取)
ls /path/to/f3set_frames/
```

**如果还没有提取帧数据**，请参考 `F3Set/README.md` 的说明提取视频帧。

### 2. 下载 ViT 预训练权重

```bash
# 下载 ViT-small 权重
wget https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
     -O ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth

# 或者使用下载脚本
python download_pretrained_vit.py --model_size small
```

---

## 🚀 训练 Stage 1

### 推荐配置 (ViT-small)

```bash
python train_vtn_stage1.py \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/f3set_frames \
    --save_dir ./vtn_outputs/stage1_small \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 50 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --pretrained_path ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth \
    --early_stop_patience 10
```

**预期输出**：

```
================================================================================
VTN Stage 1: Pre-training on F3Set
================================================================================

Loading classes from: F3Set/data/f3set-tennis/elements.txt
Loaded 30 classes

Creating datasets...
Dataset size: 2604 clips per epoch

Creating VTN model...
  Spatial size: small
  Temporal type: longformer
  Clip length: 96
  Crop dimension: 224

================================================================================
Starting training...
================================================================================

Epoch 1/50
--------------------------------------------------------------------------------
Training: 100%|███████████| 651/651 [15:23<00:00,  1.42s/it, loss=2.1234, coarse_acc=85.23%, fine_acc=42.15%]
Validation: 100%|█████████| 163/163 [03:45<00:00,  1.38s/it, loss=1.8765, coarse_acc=88.45%, fine_acc=48.32%]

Epoch 1 Results:
  Train Loss: 2.1234, Coarse Acc: 85.23%, Fine Acc: 42.15%
  Val Loss:   1.8765, Coarse Acc: 88.45%, Fine Acc: 48.32%
  ✓ Best model saved (val_loss: 1.8765)
```

**训练时间**：
- **ViT-small**: 约 8-12 小时 (4x RTX 3090)
- **ViT-base**: 约 12-18 小时 (显存需求更高)

---

## 📊 监控训练进度

### 查看训练历史

```python
import json

# 读取训练历史
with open('./vtn_outputs/stage1_small/history.json', 'r') as f:
    history = json.load(f)

# 查看最佳 epoch
best_epoch = min(history, key=lambda x: x['val_loss'])
print(f"Best Epoch: {best_epoch['epoch']}")
print(f"  Val Loss: {best_epoch['val_loss']:.4f}")
print(f"  Val Coarse Acc: {best_epoch['val_coarse_acc']*100:.2f}%")
print(f"  Val Fine Acc: {best_epoch['val_fine_acc']*100:.2f}%")
```

### 恢复中断的训练

如果训练中断，可以从最后一个 checkpoint 恢复：

```bash
python train_vtn_stage1.py \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/f3set_frames \
    --save_dir ./vtn_outputs/stage1_small \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 50 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --resume_checkpoint ./vtn_outputs/stage1_small/checkpoint_025.pt
```

---

## ✅ 验证 Stage 1 训练结果

### 检查输出文件

```bash
ls -lh ./vtn_outputs/stage1_small/
# 应该包含:
# - best_model.pt           (最佳模型，用于 Stage 3)
# - checkpoint_049.pt       (最后一个 epoch)
# - config.json             (训练配置)
# - history.json            (训练历史)
```

### 加载 Stage 1 模型

```python
import torch

# 加载 best model
checkpoint = torch.load('./vtn_outputs/stage1_small/best_model.pt')
print(f"Stage 1 Best Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Validation Coarse Acc: {checkpoint['val_coarse_acc']*100:.2f}%")
print(f"Validation Fine Acc: {checkpoint['val_fine_acc']*100:.2f}%")
```

---

## 🎯 下一步：Stage 3 微调

使用 Stage 1 checkpoint 进行 Stage 3 微调：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/stage3_from_stage1 \
    --stage1_checkpoint ./vtn_outputs/stage1_small/best_model.pt \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --learning_rate 0.0001 \
    --early_stop_patience 20
```

**注意**：
- `--stage1_checkpoint`: 指定 Stage 1 的 best model
- `--learning_rate 0.0001`: 比 Stage 1 更小的学习率（微调）

---

## ❓ 常见问题

### Q1: F3Set 帧数据在哪里？

**A:** F3Set 数据集只提供了视频文件，你需要自己提取帧。参考：

```bash
# 提取帧脚本 (示例)
python MD-FED/util/extract_frames.py \
    --video_dir /path/to/f3set_videos \
    --output_dir /path/to/f3set_frames \
    --fps 25
```

### Q2: 显存不足怎么办？

**A:**
- 减小 `--batch_size` (尝试 2 或 1)
- 使用 `--vtn_spatial_size small` 而非 `base`
- 减小 `--clip_len` (尝试 48)

### Q3: 训练太慢了，可以跳过吗？

**A:** 可以，但性能可能会差很多：

```bash
# 跳过 Stage 1，直接在 manual_annotations 上训练
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/stage3_from_scratch \
    --crop_dim 224 \
    --no_pretrained
```

### Q4: Stage 1 应该训练多久？

**A:**
- **推荐**: 使用 early stopping，通常 30-50 个 epoch 即可
- **如果 validation loss 不再下降**: 提前停止
- **如果时间有限**: 至少训练 20 个 epoch

### Q5: 如何知道 Stage 1 训练效果好不好？

**A:** 查看 validation accuracy：
- **Coarse Acc (事件检测)**: 应该 > 85%
- **Fine Acc (元素分类)**: 应该 > 40%

如果低于这些值，可能需要：
- 增加训练 epoch
- 调整学习率
- 检查数据质量

---

## 📚 相关文档

- **完整训练流程**: [`README_VTN_Complete_Training.md`](README_VTN_Complete_Training.md)
- **Stage 3 微调**: [`README_VTN_Comparison.md`](README_VTN_Comparison.md)
- **快速开始**: [`QUICKSTART_VTN.md`](QUICKSTART_VTN.md)
- **预训练权重下载**: [`FIX_HTTP_403_ERROR.md`](FIX_HTTP_403_ERROR.md)
