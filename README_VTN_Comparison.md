# VTN 对比实验指南

## 📋 概述

本指南介绍如何使用 **VTN (Video Transformer Network)** 进行对比实验，与现有的 MD-FED Stage 3 模型进行性能对比。

## ⚠️ 重要：训练与对比策略

### 训练方法

**VTN 需要在你的 manual_annotations.json 上训练**，而不是直接用预训练模型推理！

- ✅ **Spatial backbone (ViT)**: 使用 ImageNet 预训练权重（自动下载）
- ✅ **Temporal transformer**: 从随机初始化开始
- ✅ **整体模型**: 在 manual_annotations.json 上端到端训练

这与 MD-FED Stage 3 是**公平对比**：
- MD-FED Stage 3: Stage 2 预训练 + manual_annotations fine-tune
- VTN: ImageNet 预训练 + manual_annotations 训练

详细说明请参考 [`README_VTN_Training_Strategy.md`](README_VTN_Training_Strategy.md)

### 数据增强策略

为确保**公平对比**，VTN 使用与你现有方法**完全相同**的数据增强：

- ✅ **训练时**：从 398×224 帧**随机裁剪** 224×224 (`is_eval=False`)
- ✅ **评估时**：从 398×224 帧**中心裁剪** 224×224 (`is_eval=True`)

详细说明请参考 [`README_Fair_Comparison.md`](README_Fair_Comparison.md)

## 🔍 VTN vs MD-FED 对比

| 特性 | MD-FED (Stage 3) | VTN |
|------|------------------|-----|
| **Visual Backbone** | RegNetY-002 + TSM | Vision Transformer (ViT) |
| **Temporal Modeling** | GRU | Longformer/Linformer/Transformer |
| **输入模态** | RGB + Flow + Skeleton | RGB only |
| **参数量** | ~10M | ~86M (base) |
| **优势** | 轻量级、多模态融合 | 强大的自注意力机制 |

## ⚙️ VTN 配置选项

### 1. Spatial Transformer Size

```bash
--vtn_spatial_size tiny    # ViT-Tiny (192 dim, 5.7M params)
--vtn_spatial_size small   # ViT-Small (384 dim, 22M params)
--vtn_spatial_size base    # ViT-Base (768 dim, 86M params) [推荐]
--vtn_spatial_size large   # ViT-Large (1024 dim, 304M params)
```

### 2. Temporal Transformer Type

```bash
--vtn_temporal_type longformer    # Longformer (适合长序列) [推荐]
--vtn_temporal_type linformer     # Linformer (线性复杂度)
--vtn_temporal_type transformer   # 标准 Transformer
```

### 3. Temporal Architecture

```bash
--temporal_arch gru          # Single-layer GRU [推荐]
--temporal_arch deeper_gru   # Two-layer GRU
```

## 🚀 快速开始

### 1. 基础训练 (推荐配置)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/base_longformer \
    --vtn_spatial_size base \
    --vtn_temporal_type longformer \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
```

### 2. 轻量级配置 (更快训练)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/tiny_transformer \
    --vtn_spatial_size tiny \
    --vtn_temporal_type transformer \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0001
```

### 3. 高性能配置 (需要更多GPU内存)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/large_longformer \
    --vtn_spatial_size large \
    --vtn_temporal_type longformer \
    --num_epochs 50 \
    --batch_size 2 \
    --learning_rate 0.00005
```

## 📊 评估 VTN 模型

训练完成后，使用现有的评估脚本：

```bash
# 使用 evaluate_per_video.py 分别评估每个视频
python evaluate_per_video.py \
    --checkpoint_dir ./vtn_outputs/base_longformer \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --output_file vtn_per_video_results.json
```

## 🔬 对比实验设置

### 实验 1: 不同 Spatial Size 对比

```bash
# ViT-Tiny
python train_vtn_comparison.py --vtn_spatial_size tiny --save_dir ./vtn_outputs/exp1_tiny

# ViT-Small
python train_vtn_comparison.py --vtn_spatial_size small --save_dir ./vtn_outputs/exp1_small

# ViT-Base
python train_vtn_comparison.py --vtn_spatial_size base --save_dir ./vtn_outputs/exp1_base
```

### 实验 2: 不同 Temporal Type 对比

```bash
# Longformer
python train_vtn_comparison.py --vtn_temporal_type longformer --save_dir ./vtn_outputs/exp2_longformer

# Linformer
python train_vtn_comparison.py --vtn_temporal_type linformer --save_dir ./vtn_outputs/exp2_linformer

# Transformer
python train_vtn_comparison.py --vtn_temporal_type transformer --save_dir ./vtn_outputs/exp2_transformer
```

### 实验 3: VTN vs MD-FED

```bash
# 1. 训练 VTN (推荐配置)
python train_vtn_comparison.py \
    --vtn_spatial_size base \
    --vtn_temporal_type longformer \
    --save_dir ./vtn_outputs/vtn_base

# 2. 使用 MD-FED Stage 3 (已训练)
# 模型位置: ./MD-FED/md_fed_outputs/stage3

# 3. 对比评估
python evaluate_per_video.py \
    --checkpoint_dir ./vtn_outputs/vtn_base \
    --output_file vtn_results.json

python evaluate_per_video.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage3 \
    --output_file stage3_results.json

# 4. 使用 evaluate_annotations.py 计算详细指标
python evaluate_annotations.py \
    --pred_annotations vtn_annotations.json \
    --gt_annotations manual_annotations.json \
    --output vtn_evaluation.json
```

## 📈 预期结果

### 优势

1. **VTN 优势**：
   - 更强的视觉特征提取能力（ViT）
   - 长距离依赖建模（Transformer）
   - 端到端学习

2. **MD-FED 优势**：
   - 轻量级、训练更快
   - 多模态融合（RGB + Flow + Skeleton）
   - 领域特定优化

### 潜在问题

1. **VTN 可能遇到的问题**：
   - 需要更多GPU内存
   - 训练时间更长
   - 小数据集上可能过拟合

2. **解决方案**：
   - 使用较小的模型 (tiny/small)
   - 增加数据增强
   - 早停（early stopping）
   - 降低学习率

## 🛠️ 故障排除

### 1. GPU 内存不足

```bash
# 减小 batch size
--batch_size 2

# 使用更小的模型
--vtn_spatial_size tiny

# 减小 clip length
--clip_len 64
```

### 2. 训练不收敛

```bash
# 降低学习率
--learning_rate 0.00005

# 增加 warm-up epochs (在代码中修改)
warm_up_epochs = 5

# 使用更简单的 temporal type
--vtn_temporal_type transformer
```

### 3. 过拟合

```bash
# 增加训练数据比例
--train_ratio 0.9

# 早停 (在代码中添加)
# 或者减少 epochs
--num_epochs 30
```

## 📝 关键参数说明

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--vtn_spatial_size` | base | ViT 大小 | base (平衡) |
| `--vtn_temporal_type` | longformer | 时序建模 | longformer (长序列) |
| `--clip_len` | 96 | 片段长度 | 96 (与Stage3一致) |
| `--crop_dim` | 112 | 图像尺寸 | 112 (与MD-FED一致) |
| `--batch_size` | 4 | 批大小 | 4 (base), 8 (tiny) |
| `--learning_rate` | 0.0001 | 学习率 | 0.0001 |
| `--num_epochs` | 50 | 训练轮数 | 50 |

## 🎯 最佳实践

1. **先跑小模型验证**：
   ```bash
   python train_vtn_comparison.py --vtn_spatial_size tiny --num_epochs 10
   ```

2. **监控训练过程**：
   - 查看 `loss.json` 文件
   - 关注 train/val loss 的变化
   - 如果 val loss 不降，考虑早停

3. **对比实验设计**：
   - 固定数据集和划分（train_ratio=0.8）
   - 使用相同的评估指标
   - 记录训练时间和GPU使用情况

4. **结果分析**：
   - 对比 Edit Score、F1 (LCL)、F1 (event)、F1 (element)
   - 分析每个视频的单独表现
   - 查看错误序列文件

## 🔄 与 Stage 3 的完整对比流程

### Step 1: 确保有 MD-FED Stage 3 baseline

```bash
# 如果还没有 Stage 3 结果，先训练 baseline
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3_baseline
```

### Step 2: 训练 VTN

```bash
# 在相同数据上训练 VTN
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

### Step 3: 对比整体结果

```bash
# 使用对比脚本
python compare_results.py \
    --vtn_results ./vtn_outputs/comparison/best_model_metrics.json \
    --mdfed_results ./MD-FED/md_fed_outputs/stage3_baseline/evaluation_results.json \
    --output model_comparison.json
```

**预期输出**：

```
================================================================================
VTN vs MD-FED Stage 3 对比
================================================================================

指标                 MD-FED Stage 3       VTN                  差异            胜者      
─────────────────────────────────────────────────────────────────────────────────────
F1 (LCL)             0.850000            0.823000            -0.027000 (-3.2%) MD-FED    
F1 (element)         0.780000            0.795000            +0.015000 (+1.9%) VTN       
F1 (event)           0.720000            0.710000            -0.010000 (-1.4%) MD-FED    
Edit Score           0.650000            0.680000            +0.030000 (+4.6%) VTN       
─────────────────────────────────────────────────────────────────────────────────────

总结
================================================================================

总体胜率:
  VTN 胜出: 2/4 项指标 (50.0%)
  MD-FED 胜出: 2/4 项指标 (50.0%)
  持平: 0/4 项指标

推荐结论:
  ⚖️  两个模型表现相当
```

### Step 4: (可选) 分析每个视频的表现

```bash
# 分别评估每个视频 (VTN)
python evaluate_per_video.py \
    --checkpoint_dir ./vtn_outputs/comparison \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --output_file vtn_per_video_results.json

# 分别评估每个视频 (Stage 3)
python evaluate_per_video.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage3_baseline \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --output_file stage3_per_video_results.json
```

## 📚 参考资料

- VTN 论文: "Video Transformer Network" (2021)
- Vision Transformer: "An Image is Worth 16x16 Words" (ICLR 2021)
- Longformer: "Longformer: The Long-Document Transformer" (2020)
- MD-FED: 项目中的 Stage 3 实现

## ⚠️ 注意事项

1. **当前状态**：VTN 已在项目中实现但未完全集成到训练流程
2. **依赖库**：需要安装 `timm`, `einops`, `transformers`
3. **GPU要求**：base 模型至少需要 16GB GPU 内存
4. **训练时间**：VTN 比 MD-FED 慢约 2-3倍

---

**现在你可以直接运行 VTN 对比实验了！** 🚀
