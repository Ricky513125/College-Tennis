# RGB + Flow Fusion 消融实验说明

## 📋 消融实验的目的

这个实验是为了测试**不使用 Skeleton 模态**时，仅用 **RGB + Flow** 的效果如何。

### 对比设置

| 模型 | 输入模态 | 预训练来源 | 目的 |
|------|---------|-----------|------|
| **MD-FED Stage 3** | RGB + Flow + **Skeleton** | Stage 2 (多模态蒸馏) | 完整模型（baseline） |
| **RGB + Flow Fusion** | RGB + Flow (无 Skeleton) | Stage 2 (多模态蒸馏) | **消融实验**：测试 Skeleton 的重要性 |

### 实验意义

通过对比这两个模型，可以回答：
- **Skeleton 模态对最终性能的贡献有多大？**
- 如果只有 RGB + Flow，性能会下降多少？
- 这是否证明了多模态融合（特别是 Skeleton）的必要性？

---

## ⚠️ 为什么效果可能很差？

### 主要原因

1. **缺少 Stage 2 预训练（最关键）** ⭐
   - MD-FED Stage 3 使用了 **Stage 2 的预训练权重**（多模态蒸馏）
   - 如果 RGB+Flow Fusion 只用了 ImageNet 预训练，没有经过 Stage 2 蒸馏，效果会差很多
   - **解决方案**：使用 `--stage2_checkpoint` 参数加载 MD-FED Stage 2 的权重

2. **缺少 Skeleton 信息**
   - Skeleton（姿态）对于细粒度动作识别很重要
   - 这是消融实验的预期结果：证明 Skeleton 的重要性

3. **训练策略不一致**
   - 确保使用相同的学习率、batch size、数据增强等

---

## 🚀 正确的训练方法

### 方法 1：使用 Stage 2 预训练（推荐）⭐⭐⭐

**这是与 MD-FED Stage 3 进行公平对比的正确方法！**

```bash
python train_rgb_flow_fusion.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /path/to/optical_flows \
    --save_dir ./rgb_flow_fusion_outputs \
    --stage2_checkpoint ./md_fed_outputs/stage2/checkpoint_XXX.pt \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_epochs 500 \
    --visual_arch rny002_tsm \
    --temporal_arch gru \
    --fusion_method add \
    --learning_rate 0.0001 \
    --early_stop_patience 20
```

**关键点**：
- ✅ 使用 `--stage2_checkpoint` 加载 MD-FED Stage 2 的权重
- ✅ 这样 RGB 和 Flow 特征提取器都经过了 Stage 2 的多模态蒸馏训练
- ✅ 与 MD-FED Stage 3 的对比是公平的（都使用 Stage 2 预训练）

### 方法 2：仅使用 ImageNet 预训练（不推荐）

```bash
python train_rgb_flow_fusion.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /path/to/optical_flows \
    --save_dir ./rgb_flow_fusion_outputs \
    # 不使用 --stage2_checkpoint，只使用 ImageNet 预训练
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_epochs 500 \
    --visual_arch rny002_tsm \
    --temporal_arch gru \
    --fusion_method add \
    --learning_rate 0.0001 \
    --early_stop_patience 20
```

**问题**：
- ❌ 没有经过 Stage 2 蒸馏，特征质量不如 MD-FED
- ❌ 与 MD-FED Stage 3 的对比不公平
- ❌ 效果会明显更差

---

## 📊 如何找到 Stage 2 Checkpoint？

### 方法 1：使用最佳 epoch

```bash
# 查看 Stage 2 训练历史
cat ./md_fed_outputs/stage2/history.json

# 或查看 loss.json
cat ./md_fed_outputs/stage2/loss.json

# 找到验证损失最低的 epoch，然后使用：
--stage2_checkpoint ./md_fed_outputs/stage2/checkpoint_XXX.pt
```

### 方法 2：使用 best.pt（如果存在）

```bash
--stage2_checkpoint ./md_fed_outputs/stage2/best.pt
```

### 方法 3：使用最新 epoch

```bash
# 列出所有 checkpoint
ls -t ./md_fed_outputs/stage2/checkpoint_*.pt

# 使用最新的
--stage2_checkpoint ./md_fed_outputs/stage2/checkpoint_049.pt
```

---

## 🔍 评估结果解读

### 预期结果

如果消融实验设计正确，你应该看到：

| 模型 | Edit Score | Fine Precision | Fine Recall | Fine F1 |
|------|-----------|----------------|-------------|---------|
| **MD-FED Stage 3** (RGB+Flow+Skeleton) | ~X.XX | ~X.XX | ~X.XX | ~X.XX |
| **RGB + Flow Fusion** (无 Skeleton) | **< X.XX** | **< X.XX** | **< X.XX** | **< X.XX** |

**预期**：RGB + Flow Fusion 的性能应该**低于** MD-FED Stage 3，这证明了：
- ✅ Skeleton 模态的重要性
- ✅ 多模态融合的必要性

### 如果效果太差（异常）

如果 RGB + Flow Fusion 的性能**远低于预期**（例如 Edit Score < 0.1），可能的原因：

1. **没有使用 Stage 2 预训练**
   - 检查是否使用了 `--stage2_checkpoint`
   - 如果没有，这是主要原因

2. **Stage 2 checkpoint 加载失败**
   - 检查日志中是否有 "Successfully loaded Stage 2 checkpoint"
   - 检查加载的参数数量是否合理

3. **训练不充分**
   - 增加 `--num_epochs`
   - 检查验证损失是否还在下降

4. **学习率不合适**
   - 尝试更小的学习率（如 `0.00005`）
   - 或使用学习率调度器

---

## 📝 实验报告建议

在论文中报告消融实验结果时，建议包括：

1. **实验设置**
   - 明确说明使用了 Stage 2 预训练（公平对比）
   - 说明训练策略与 MD-FED Stage 3 一致

2. **结果对比**
   - 表格对比 MD-FED Stage 3 vs RGB + Flow Fusion
   - 分析性能下降的幅度

3. **结论**
   - Skeleton 模态对性能的贡献（例如：提升 X% 的 Edit Score）
   - 证明多模态融合的必要性

---

## 🛠️ 故障排除

### 问题 1：Stage 2 checkpoint 加载失败

**错误信息**：
```
⚠️  Warning: Could not load any weights from Stage 2 checkpoint
```

**解决方案**：
1. 检查 checkpoint 路径是否正确
2. 检查 checkpoint 文件是否完整
3. 尝试使用不同的 epoch 的 checkpoint

### 问题 2：训练损失不下降

**可能原因**：
- 学习率太大或太小
- 没有正确加载 Stage 2 权重
- 数据加载有问题

**解决方案**：
1. 检查是否成功加载了 Stage 2 权重（查看日志）
2. 调整学习率
3. 检查数据加载是否正常

### 问题 3：验证损失为 NaN

**可能原因**：
- 类别权重太大（event1 权重 20 倍可能导致数值不稳定）
- 学习率太大

**解决方案**：
1. 降低 event1 权重（例如从 20 降到 10）
2. 降低学习率
3. 使用梯度裁剪

---

## 📚 相关文档

- [`README_MD-FED_Stage2.md`](README_MD-FED_Stage2.md) - MD-FED Stage 2 训练指南
- [`few_shot_learning_stage3.py`](few_shot_learning_stage3.py) - MD-FED Stage 3 训练脚本
- [`evaluate_comparison_models.py`](evaluate_comparison_models.py) - 模型评估脚本
