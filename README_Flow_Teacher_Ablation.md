# Flow as Teacher 消融实验说明

## 📋 消融实验的目的

这个实验是为了测试**使用 Flow 作为教师网络**的效果，与原始 MD-FED（Skeleton 作为教师）和 RGB as Teacher 进行对比。

### 对比设置

| 实验 | 教师网络 | 学生网络 | 蒸馏损失 |
|------|---------|---------|---------|
| **原始 MD-FED Stage 2** | **Skeleton** | RGB, Flow | `MSE(RGB_feat, Skeleton_feat) + MSE(Flow_feat, Skeleton_feat)` |
| **RGB as Teacher (消融)** | **RGB** | Flow, Skeleton | `MSE(Flow_feat, RGB_feat) + MSE(Skeleton_feat, RGB_feat)` |
| **Flow as Teacher (消融)** | **Flow** | RGB, Skeleton | `MSE(RGB_feat, Flow_feat) + MSE(Skeleton_feat, Flow_feat)` |

### 实验意义

通过对比这三个实验，可以回答：
- **不同模态作为教师网络的效果如何？**
- Flow 作为教师是否比 Skeleton 或 RGB 作为教师更有效？
- 这是否证明了 Skeleton 作为教师的优势？
- Flow 特征是否包含足够的运动信息来指导其他模态？

---

## 🚀 训练步骤

### 步骤 1: 准备 Stage 1 模型（Skeleton 预训练）

首先需要训练 Stage 1，为 Skeleton 网络提供预训练权重：

```bash
python run_md_fed_stage1.py \
    --pose_dir /path/to/skeletons/f3set-tennis \
    --output_dir ./md_fed_outputs/stage1 \
    --num_epochs 500 \
    --batch_size 4 \
    --learning_rate 0.001
```

### 步骤 2: 训练 Stage 2 (Flow as Teacher)

使用新创建的脚本进行训练：

```bash
python train_stage2_flow_teacher.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --pose_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_skeleton_annotations_rally \
    --stage1_model_dir ./MD-FED/md_fed_outputs/stage1 \
    --save_dir ./md_fed_outputs/stage2_flow_teacher \
    --data_dir ./md_fed_data \
    --dataset_name ncaa-rally \
    --visual_arch rny002_tsm \
    --skeleton_arch stgcn++ \
    --temporal_arch gru \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --clip_len 96 \
    --stride 2 \
    --crop_dim 224
```

### 步骤 3: 训练 Stage 3 (Few-shot Fine-tuning)

使用 Stage 2 的权重进行 Stage 3 训练：

```bash
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./md_fed_outputs/stage2_flow_teacher \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./md_fed_outputs/stage3_flow_teacher \
    --num_epochs 500 \
    --batch_size 4 \
    --learning_rate 0.0001
```

---

## 🔍 关键区别

### 原始 MD-FED Stage 2

```python
# Skeleton 作为教师
rgb2sk_loss = MSE(rgb_feat, sk_feat)
flow2sk_loss = MSE(flow_feat, sk_feat)
loss = rgb2sk_loss + flow2sk_loss
```

**逻辑**：
- Skeleton 特征已经通过 Stage 1 预训练，具有丰富的动作表示
- RGB 和 Flow 学习 Skeleton 的特征表示
- 这样 RGB 和 Flow 可以学到 Skeleton 的知识

### RGB as Teacher (消融实验)

```python
# RGB 作为教师
flow2rgb_loss = MSE(flow_feat, rgb_feat)
sk2rgb_loss = MSE(sk_feat, rgb_feat)
loss = flow2rgb_loss + sk2rgb_loss
```

**逻辑**：
- RGB 特征来自 ImageNet 预训练的视觉模型
- Flow 和 Skeleton 学习 RGB 的特征表示
- 测试视觉特征作为教师的效果

### Flow as Teacher (消融实验)

```python
# Flow 作为教师
rgb2flow_loss = MSE(rgb_feat, flow_feat)
sk2flow_loss = MSE(sk_feat, flow_feat)
loss = rgb2flow_loss + sk2flow_loss
```

**逻辑**：
- Flow 特征来自光流数据，包含运动信息
- RGB 和 Skeleton 学习 Flow 的特征表示
- 测试运动特征作为教师的效果

---

## 📊 预期结果

### 如果 Flow 作为教师效果最好

- ✅ 说明运动特征（Flow）作为教师最有效
- ✅ 可能因为 Flow 直接编码了运动信息，更适合动作识别
- ❌ 但这与 MD-FED 的设计理念相矛盾（Skeleton 更适合动作识别）

### 如果 Skeleton 作为教师效果最好（预期）

- ✅ 证明 MD-FED 的设计是正确的
- ✅ Skeleton 特征更适合作为动作识别的教师
- ✅ 说明姿态信息对于动作理解的重要性

### 如果 RGB 作为教师效果最好

- ✅ 说明视觉特征（RGB）作为教师更有效
- ✅ 可能因为 RGB 包含更丰富的视觉信息
- ❌ 但这与 MD-FED 的设计理念相矛盾

---

## ⚙️ 训练参数说明

### 必需参数

- `--manual_annotations`: 标注文件路径
- `--frame_dir`: RGB 帧目录
- `--flow_dir`: 光流文件目录
- `--pose_dir`: Skeleton 文件目录
- `--stage1_model_dir`: Stage 1 模型目录（用于初始化 Skeleton）
- `--save_dir`: 保存 Stage 2 模型的目录

### 可选参数

- `--visual_arch`: 视觉架构（默认：`rny002_tsm`）
- `--skeleton_arch`: Skeleton 架构（默认：`stgcn++`）
- `--batch_size`: Batch size（默认：4）
- `--num_epochs`: 训练轮数（默认：50）
- `--learning_rate`: 学习率（默认：0.001）
- `--clip_len`: 视频片段长度（默认：96）
- `--stride`: 帧步长（默认：2）

---

## 🔬 实验对比

完成训练后，可以对比以下结果：

| 模型 | Stage 2 教师 | Stage 3 Edit Score | Stage 3 Fine F1 |
|------|-------------|-------------------|-----------------|
| **MD-FED (原始)** | Skeleton | X.XX | X.XX |
| **RGB as Teacher** | RGB | X.XX | X.XX |
| **Flow as Teacher** | Flow | X.XX | X.XX |

**分析**：
- 如果 Flow as Teacher 效果最好，说明运动特征作为教师更有效
- 如果 MD-FED 效果最好，说明 Skeleton 作为教师的设计是正确的
- 如果 RGB as Teacher 效果最好，说明视觉特征作为教师更有效

---

## ⚠️ 注意事项

1. **Stage 1 预训练是必需的**
   - 即使 Flow 作为教师，Skeleton 网络仍然需要 Stage 1 预训练
   - 这确保了 Skeleton 网络有合理的初始化

2. **训练时间**
   - Stage 2 训练通常需要 50 个 epoch
   - 每个 epoch 的时间取决于数据集大小和 batch size

3. **内存使用**
   - 如果遇到 OOM 错误，可以：
     - 减小 `--batch_size`
     - 减小 `--clip_len`
     - 增加 `--acc_grad_iter`（梯度累积）

4. **数据要求**
   - 需要 RGB 帧、光流和 Skeleton 数据
   - 数据格式与原始 MD-FED 相同

---

## 📝 实验报告建议

在论文中报告消融实验结果时，建议包括：

1. **实验设置**
   - 明确说明 Flow 作为教师的设置
   - 说明训练策略与原始 MD-FED 一致

2. **结果对比**
   - 表格对比三种教师网络的效果
   - 分析性能差异的原因

3. **结论**
   - 证明 Skeleton 作为教师的优势（如果结果支持）
   - 或讨论 Flow/RGB 作为教师的可能性

---

## 🛠️ 故障排除

### 问题 1: Stage 1 checkpoint 加载失败

**错误信息**：
```
⚠️  Warning: Stage 1 checkpoint not found
```

**解决方案**：
1. 确保 Stage 1 训练已完成
2. 检查 `--stage1_model_dir` 路径是否正确
3. 确保目录中包含 `checkpoint_XXX.pt` 文件

### 问题 2: 训练损失不下降

**可能原因**：
- 学习率太大或太小
- Flow 特征与 RGB/Skeleton 特征维度不匹配

**解决方案**：
1. 调整学习率（尝试 0.0005 或 0.002）
2. 检查特征维度是否一致
3. 查看训练日志中的损失值

### 问题 3: 内存不足

**解决方案**：
1. 减小 batch size
2. 减小 clip_len
3. 使用梯度累积（`--acc_grad_iter 2`）

---

## 📚 相关文档

- [`README_RGB_Teacher_Ablation.md`](README_RGB_Teacher_Ablation.md) - RGB as Teacher 消融实验指南
- [`README_MD-FED_Stage2.md`](README_MD-FED_Stage2.md) - 原始 MD-FED Stage 2 训练指南
- [`train_md_fed_stage2.py`](train_md_fed_stage2.py) - 原始 Stage 2 训练脚本
- [`few_shot_learning_stage3.py`](few_shot_learning_stage3.py) - Stage 3 训练脚本
