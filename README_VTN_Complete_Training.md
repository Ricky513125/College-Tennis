# VTN 训练策略：跳过 Stage 1 的合理性

## ⚠️ 重要澄清

**VTN 应该跳过 Stage 1，直接在 manual_annotations.json 上训练！**

## 📋 为什么跳过 Stage 1？

### MD-FED 的三阶段训练

| 阶段 | 输入数据 | 模型组件 | 目的 |
|------|---------|---------|------|
| **Stage 1** | **Skeleton (pose)** | STGCN++ | Skeleton 特征预训练 |
| **Stage 2** | RGB + Flow + Skeleton | RGB/Flow → Skeleton 蒸馏 | 多模态融合 |
| **Stage 3** | RGB + Flow | 完整模型 | Few-shot 微调 |

### VTN 的训练策略

| 阶段 | 输入数据 | 模型组件 | 目的 |
|------|---------|---------|------|
| **Stage 1** | ❌ **跳过** | - | VTN 不使用 skeleton |
| **Stage 2** | ❌ **跳过** | - | VTN 是单模态（纯 RGB） |
| **Stage 3** | RGB only | ViT + Longformer | Few-shot 微调 |

### 关键区别

1. **MD-FED Stage 1**: 使用 **skeleton 数据** + STGCN++
2. **VTN**: 不使用 skeleton，只用 RGB

**结论**: VTN 在 RGB 上做 Stage 1 是不合理的，因为：
- 输入模态与 MD-FED 完全不同（RGB vs Skeleton）
- 对比会不公平（不同的预训练数据）

---

## 🚀 推荐训练步骤

### **直接在 manual_annotations.json 上训练**

使用 ImageNet 预训练的 ViT，直接在你的数据上训练：

#### **推荐配置 (ViT-small) - 快速验证**

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_small \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --pretrained_path ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth \
    --early_stop_patience 20
```

**训练时间**: 约 2-4 小时

#### **高性能配置 (ViT-base) - 最佳性能**

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_base \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 2 \
    --num_epochs 500 \
    --vtn_spatial_size base \
    --vtn_temporal_type longformer \
    --pretrained_path ~/.cache/torch/hub/checkpoints/vit_base_patch16_224.pth \
    --early_stop_patience 20
```

**训练时间**: 约 4-8 小时（显存需求更高）

---

### **评估 VTN 模型**

使用训练好的模型进行评估：

```bash
python evaluate_comparison_models.py \
    --model_type vtn \
    --checkpoint ./vtn_outputs/vtn_base/best_model.pt \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --vtn_spatial_size base \
    --vtn_temporal_type longformer \
    --clip_len 96 \
    --crop_dim 224 \
    --save_predictions
```

---

### **与 MD-FED 对比**

#### **评估 MD-FED Stage 3 模型**

```bash
python evaluate_comparison_models.py \
    --model_type md_fed \
    --checkpoint ./MD-FED/md_fed_outputs/stage3/best_checkpoint.pt \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt
```

#### **对比结果**

```bash
python compare_results.py \
    ./vtn_outputs/stage3_from_stage1/evaluation_results.json \
    ./MD-FED/md_fed_outputs/stage3/evaluation_results.json
```

---

## 📊 预期性能对比

| 模型 | Edit Score | F1 (Fine) | 训练时间 (Stage 1) | 训练时间 (Stage 3) |
|------|------------|-----------|-------------------|-------------------|
| **MD-FED** | ~0.85 | ~0.80 | 8h (skeleton) | 2-4h |
| **VTN (from Stage 1)** | ? | ? | 8-12h (RGB) | 2-4h |
| **VTN (from scratch)** | ? | ? | - | 6-12h |

---

## 🔍 关键区别：MD-FED vs VTN

| 特性 | MD-FED | VTN |
|------|--------|-----|
| **Stage 1 输入** | Skeleton (2D pose) | RGB frames |
| **Stage 1 模型** | GCN + GRU | ViT + Longformer |
| **Stage 2** | 多模态蒸馏 | ❌ 跳过 |
| **Stage 3 输入** | RGB + Flow | RGB only |
| **Stage 3 初始化** | From Stage 2 | From Stage 1 |

---

## ⚠️ 常见问题

### **Q1: 为什么不用 Flow 和 Skeleton？**

**A:** VTN 是纯 Vision Transformer 模型，设计上只使用 RGB 输入。添加 Flow/Skeleton 需要重新设计模型架构。

### **Q2: 跳过 Stage 2 是否公平？**

**A:** 是的。MD-FED 的 Stage 2 是为了多模态融合，而 VTN 是单模态模型，不需要这一步。

### **Q3: Stage 1 应该训练多少个 epoch？**

**A:** 推荐 50 个 epoch，并使用 early stopping。如果 validation loss 不再下降，可以提前停止。

### **Q4: Stage 3 的学习率应该设置多少？**

**A:**
- **从 Stage 1 开始**：使用 `0.0001` (更小的学习率用于微调)
- **从随机初始化**：使用 `0.001` (更大的学习率)

### **Q5: 显存不足怎么办？**

**A:**
- 减小 `--batch_size` (推荐 1 或 2)
- 使用 `--vtn_spatial_size small` 而非 `base`
- 减小 `--clip_len` (例如 48 而非 96)
- 减小 `--crop_dim` (例如 112 而非 224)

### **Q6: 如何确认 Stage 1 checkpoint 加载成功？**

**A:** 查看训练日志，应该看到：

```
Loading Stage 1 checkpoint: ./vtn_outputs/stage1_small/best_model.pt
✓ Successfully loaded Stage 1 checkpoint
  Loaded parameters: 150/152
```

如果 `Loaded parameters` 接近总参数数量，说明加载成功。

---

## 📁 目录结构

完整训练后的目录结构：

```
vtn_outputs/
├── stage1_small/               # Stage 1: F3Set 预训练
│   ├── best_model.pt           # 最佳模型 (用于 Stage 3 初始化)
│   ├── checkpoint_049.pt       # 最后一个 epoch
│   ├── config.json             # 训练配置
│   └── history.json            # 训练历史
│
└── stage3_from_stage1/         # Stage 3: Few-shot 微调
    ├── best_model.pt           # 最佳模型 (用于评估)
    ├── checkpoint_499.pt       # 最后一个 epoch
    ├── train/                  # 训练数据
    │   └── train.json
    ├── val/                    # 验证数据
    │   └── val.json
    └── evaluation_results.json # 评估结果
```

---

## 📚 相关文档

- **快速开始**: `QUICKSTART_VTN.md`
- **完整对比流程**: `QUICKSTART_Comparison_Models.md`
- **VTN 基础配置**: `README_VTN_Comparison.md`
- **数据增强策略**: `README_Fair_Comparison.md`
- **矩形输入支持**: `README_Rectangular_Input.md`

---

## 🎯 总结

1. **Stage 1 (必需)**: 在 F3Set 上预训练，获得通用的网球动作特征
2. **Stage 2 (跳过)**: VTN 不需要多模态蒸馏
3. **Stage 3 (必需)**: 从 Stage 1 初始化，在 manual_annotations 上微调

这样的训练流程能够确保 VTN 和 MD-FED 在相同的起点上进行对比，是公平的比较方案！🎾
