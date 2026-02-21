# RGB + Flow + Skeleton 直接融合消融实验说明

## 📋 消融实验的目的

这个实验是为了测试**直接融合 RGB、Flow 和 Skeleton 三种模态**的效果，**不使用蒸馏（distillation）**。

根据论文描述：
> "fusing RGB, optical flow, and skeleton leads to significantly lower performance, 
> with edit scores dropping by up to 21.8%, highlighting the superiority of 
> distillation over direct fusion"

### 对比设置

| 模型 | 输入模态 | 训练方法 | 目的 |
|------|---------|---------|------|
| **MD-FED Stage 3** | RGB + Flow + Skeleton | Stage 2 多模态蒸馏 | 完整模型（baseline） |
| **RGB+Flow+Skeleton Fusion** | RGB + Flow + Skeleton | 直接融合（无蒸馏） | **消融实验**：对比直接融合 vs 蒸馏 |

### 实验意义

通过对比这两个模型，可以回答：
- **蒸馏方法是否比直接融合更有效？**
- 直接融合三种模态会导致性能下降多少？
- 这是否证明了 MD-FED 中蒸馏策略的必要性？

---

## 🚀 训练步骤

### Step 1: Stage 1 预训练（Skeleton）

首先需要预训练 Skeleton 特征提取器：

```bash
python train_stgcn_stage1.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --pose_dir /path/to/ncaa_skeletons_rally \
    --save_dir ./md_fed_outputs/stage1 \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --learning_rate 0.001
```

### Step 2: 训练 RGB+Flow+Skeleton 直接融合模型

```bash
python train_rgb_flow_skeleton_fusion.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --flow_dir /path/to/ncaa_flow_rally \
    --pose_dir /path/to/ncaa_skeletons_rally \
    --save_dir ./rgb_flow_skeleton_fusion_outputs \
    --stage1_model_dir ./md_fed_outputs/stage1 \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --learning_rate 0.001 \
    --fusion_method concat \
    --gradient_accumulation_steps 1 \
    --early_stop_patience 20
```

### 可选：使用 Stage 2 预训练（公平对比）

为了与 MD-FED Stage 3 进行公平对比，可以加载 MD-FED Stage 2 的 RGB 和 Flow 权重：

```bash
python train_rgb_flow_skeleton_fusion.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --flow_dir /path/to/ncaa_flow_rally \
    --pose_dir /path/to/ncaa_skeletons_rally \
    --save_dir ./rgb_flow_skeleton_fusion_outputs \
    --stage1_model_dir ./md_fed_outputs/stage1 \
    --stage2_checkpoint ./md_fed_outputs/stage2/best_model.pt \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500 \
    --learning_rate 0.001 \
    --fusion_method concat
```

---

## 🔧 关键参数说明

### 融合方法 (`--fusion_method`)

- **`concat`** (推荐): 拼接三种模态的特征，然后通过线性层融合
  - 公式: `fused = Linear([rgb_feat, flow_feat, sk_feat])`
  - 优点: 最灵活，可以学习不同模态的权重

- **`add`**: 直接相加三种模态的特征
  - 公式: `fused = rgb_feat + flow_feat + sk_feat`
  - 要求: 三种模态的特征维度必须相同

- **`weighted`**: 可学习的加权组合
  - 公式: `fused = w_rgb * rgb_feat + w_flow * flow_feat + w_sk * sk_feat`
  - 优点: 可以自动学习各模态的重要性

### 模型架构

- **Visual arch**: `rny002_tsm` (RegNetY-002 + TSM)
- **Skeleton arch**: `stgcn++` (STGCN++)
- **Temporal arch**: `gru` (单层 GRU)

---

## 📊 预期结果

根据论文描述，直接融合应该会导致：
- **Edit score 下降约 21.8%**
- 性能明显低于使用蒸馏的 MD-FED Stage 3

这证明了：
1. **蒸馏方法比直接融合更有效**
2. **MD-FED 的 Stage 2 蒸馏策略是必要的**

---

## 🔍 与 MD-FED 的关键区别

| 特性 | MD-FED Stage 2 | RGB+Flow+Skeleton Fusion |
|------|---------------|-------------------------|
| **训练目标** | 蒸馏损失 (MSE) | 分类损失 (CrossEntropy + BCE) |
| **特征融合** | 不直接融合，通过蒸馏对齐 | 直接融合三种模态特征 |
| **训练方式** | RGB/Flow 学习 Skeleton 特征 | 三种模态共同训练 |
| **Stage 3** | 使用蒸馏后的特征 | 使用融合后的特征 |

---

## ⚠️ 注意事项

1. **必须进行 Stage 1 预训练**
   - Skeleton 特征提取器需要先在 F3Set 上预训练
   - 使用 `--stage1_model_dir` 加载预训练权重

2. **公平对比**
   - 为了公平对比，建议使用 `--stage2_checkpoint` 加载 MD-FED Stage 2 的 RGB/Flow 权重
   - 这样可以确保特征提取器的初始化条件相同

3. **内存优化**
   - 如果遇到 OOM 错误，可以：
     - 减小 `--batch_size`
     - 增加 `--gradient_accumulation_steps`
     - 减小 `--clip_len`

4. **融合方法选择**
   - 推荐使用 `concat`，因为它最灵活
   - `add` 要求特征维度相同，可能不适合所有配置

---

## 📈 评估

训练完成后，可以使用 `evaluate_comparison_models.py` 进行评估：

```bash
python evaluate_comparison_models.py \
    --model_type rgb_flow_skeleton_fusion \
    --checkpoint ./rgb_flow_skeleton_fusion_outputs/best_model.pt \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --flow_dir /path/to/ncaa_flow_rally \
    --pose_dir /path/to/ncaa_skeletons_rally \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4
```

---

## 🐛 故障排除

### 问题 1: Skeleton 维度错误

**错误**: `ValueError: not enough values to unpack (expected 5, got 4)`

**解决**: 脚本已经包含了 `collate_fn_skeleton_padding` 来处理不同数量的检测人员。

### 问题 2: 内存不足

**错误**: `torch.OutOfMemoryError`

**解决**: 
- 减小 `--batch_size` (例如从 4 到 2)
- 增加 `--gradient_accumulation_steps` (例如从 1 到 2)
- 减小 `--clip_len` (例如从 96 到 64)

### 问题 3: 性能很差

**可能原因**:
1. 没有加载 Stage 1 预训练权重
2. 没有加载 Stage 2 预训练权重（用于公平对比）
3. 融合方法不适合当前配置

**解决**: 确保使用 `--stage1_model_dir` 和 `--stage2_checkpoint` 参数。

---

## 📝 参考文献

根据论文中的描述：
> "Multimodal fusion vs. distillation. To investigate the optimal integration 
> of multimodal information, we compare multimodal fusion. As shown in Table 3(a), 
> fusing RGB, optical flow, and skeleton leads to significantly lower performance, 
> with edit scores dropping by up to 21.8%, highlighting the superiority of 
> distillation over direct fusion"
