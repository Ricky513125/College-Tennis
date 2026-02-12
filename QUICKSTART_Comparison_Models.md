# 快速对比实验指南

本指南提供 MD-FED Stage 3、VTN 和 I3D 三个模型的快速训练和对比方法。

## 🎯 实验目标

在 `manual_annotations.json` 数据集上比较三个模型的性能：

1. **MD-FED Stage 3**（基线）：多模态深度学习框架
2. **VTN**：Vision Transformer + Temporal Transformer
3. **I3D**：Inflated 3D ConvNet

## 📋 前置要求

- Python 3.8+
- PyTorch with CUDA
- 预处理好的帧数据
- `manual_annotations.json`

## 🚀 快速开始

### 1. MD-FED Stage 3（基线模型）

```bash
# 训练
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3

# 评估（每个视频单独）
python evaluate_per_video.py \
    --checkpoint ./MD-FED/md_fed_outputs/stage3/best_model.pt \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally
```

### 2. VTN（Vision Transformer）

```bash
# 训练
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./vtn_outputs \
    --pretrained_path ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer

# 评估
python evaluate_per_video.py \
    --checkpoint ./vtn_outputs/best_model.pt \
    --model_type vtn \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally
```

### 3. I3D（3D CNN）

```bash
# 训练
python train_i3d_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./i3d_outputs \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500

# 评估
python evaluate_per_video.py \
    --checkpoint ./i3d_outputs/best_model.pt \
    --model_type i3d \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally
```

## 📊 模型对比

### 架构特点

| 模型 | 输入模态 | 骨干网络 | 时间建模 | 参数量 | 显存占用 |
|------|---------|---------|---------|--------|---------|
| **MD-FED Stage 3** | RGB + Flow + Skeleton | ResNet + GCN | GRU | ~15M | 中 |
| **VTN** | RGB | ViT-small | Longformer | ~22M | 中 |
| **I3D** | RGB | Inception-I3D | 3D Conv | ~12M | 高 |

### 数据增强策略

所有模型使用**相同的数据增强策略**以确保公平对比：

- **训练**：随机裁剪到 224×224
- **验证**：居中裁剪到 224×224
- **数据分割**：80% 训练，20% 验证（相同的随机种子）

### 推荐配置

#### 小数据集（< 100 样本）

```bash
# VTN (推荐)
--crop_dim 112 --batch_size 8 --num_epochs 300

# I3D
--crop_dim 112 --batch_size 4 --num_epochs 300
```

#### 中等数据集（100-1000 样本）

```bash
# VTN
--crop_dim 224 --batch_size 4 --num_epochs 500

# I3D
--crop_dim 224 --batch_size 2 --num_epochs 500
```

#### 大数据集（> 1000 样本）

```bash
# VTN
--crop_dim 224 --batch_size 8 --vtn_spatial_size base

# I3D
--crop_dim 224 --batch_size 4 --dropout 0.3
```

## 🔧 显存优化

### 显存有限（< 8GB）

```bash
# VTN
--batch_size 2 --clip_len 64 --crop_dim 112

# I3D
--batch_size 1 --clip_len 64 --crop_dim 112
```

### 显存充足（>= 16GB）

```bash
# VTN
--batch_size 8 --clip_len 96 --crop_dim 224 --vtn_spatial_size base

# I3D
--batch_size 4 --clip_len 96 --crop_dim 224
```

## 📈 训练监控

所有模型训练时都会显示：

```
Epoch 1/500 [Train]: 100%|████████| 19/19 [00:45<00:00, loss: 2.3456]
Epoch 1/500 [Val]: 100%|████████| 5/5 [00:08<00:00, loss: 2.1234]
[Epoch 1/500] Train: 2.3456 | Val: 2.1234 | LR: 1.00e-04
  ✅ New best val loss: 2.1234 → Saving checkpoint...
```

## 📁 输出结构

```
./
├── MD-FED/md_fed_outputs/stage3/
│   ├── best_model.pt
│   └── loss.json
├── vtn_outputs/
│   ├── best_model.pt
│   └── loss.json
└── i3d_outputs/
    ├── best_model.pt
    └── loss.json
```

## 🔬 结果比较

训练完成后，使用 `compare_results.py` 比较三个模型：

```bash
python compare_results.py \
    --mdfed_results ./MD-FED/md_fed_outputs/stage3/evaluation_results.json \
    --vtn_results ./vtn_outputs/evaluation_results.json \
    --i3d_results ./i3d_outputs/evaluation_results.json
```

输出示例：

```
╔════════════════════════════════════════════════════════════╗
║              Model Performance Comparison                  ║
╠════════════════════════════════════════════════════════════╣
║ Metric          │ MD-FED    │ VTN       │ I3D       │ Best  ║
╠────────────────────────────────────────────────────────────╣
║ F1 (LCL)        │ 0.8234    │ 0.8156    │ 0.7989    │ MD-FED║
║ F1 (element)    │ 0.7456    │ 0.7523    │ 0.7234    │ VTN   ║
║ F1 (event)      │ 0.8901    │ 0.8834    │ 0.8756    │ MD-FED║
║ Edit Score      │ 0.7823    │ 0.7765    │ 0.7654    │ MD-FED║
╚════════════════════════════════════════════════════════════╝
```

## ⚡ 训练时间估计

在单个 NVIDIA RTX 3090 上，每个 epoch 的训练时间：

| 模型 | Batch Size 4 | Batch Size 2 |
|------|-------------|-------------|
| **MD-FED Stage 3** | ~5 min | ~8 min |
| **VTN (small)** | ~3 min | ~5 min |
| **I3D** | ~7 min | ~12 min |

500 epochs 预计训练时间：
- **VTN**: ~25 小时
- **I3D**: ~58 小时
- **MD-FED Stage 3**: ~42 小时

## 💡 常见问题

### Q1: 哪个模型最好？

A: 取决于你的需求：
- **准确率优先**: MD-FED Stage 3（多模态）
- **速度优先**: VTN
- **经典方法**: I3D

### Q2: 必须同时训练三个吗？

A: 不必须。你可以：
1. 先训练 VTN（最快）
2. 如果效果不满意，再尝试 MD-FED 或 I3D

### Q3: 预训练权重重要吗？

A: 对于小数据集（< 100 样本）：
- **VTN**: 预训练很重要 ✅
- **I3D**: 预训练有帮助但不是必须
- **MD-FED**: 使用 Stage 2 模型初始化

### Q4: 如何选择 batch_size？

A: 根据显存决定：
```python
显存 (GB)  │  VTN  │  I3D
─────────────────────────
    8     │   2   │   1
   12     │   4   │   2
   16     │   8   │   4
   24+    │  16   │   8
```

## 🚨 故障排除

### CUDA Out of Memory

```bash
# 降低 batch_size 和 clip_len
--batch_size 2 --clip_len 64

# 或降低图像尺寸
--crop_dim 112
```

### 训练不收敛

```bash
# 增加 warmup
--warmup_epochs 20

# 降低学习率
--learning_rate 5e-5

# 增加正则化
--weight_decay 1e-3
```

### 过拟合

```bash
# 增加 dropout
--dropout 0.7

# 增加数据增强
# (在代码中修改 ActionSeqDataset)

# 使用更小的模型
--vtn_spatial_size tiny  # For VTN
```

## 📚 相关文档

- `README_VTN_Comparison.md` - VTN 详细说明
- `README_I3D_Comparison.md` - I3D 详细说明
- `README_Fair_Comparison.md` - 公平对比策略
- `README_VTN_Training_Strategy.md` - VTN 训练策略

## 🎓 引用

如果这些模型对你的研究有帮助，请引用原始论文：

```bibtex
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition?},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={CVPR},
  year={2017}
}

@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}
```
