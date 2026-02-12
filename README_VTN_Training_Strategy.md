# VTN 训练策略与对比方法

## 🤔 核心问题

**VTN 应该如何训练？如何与 MD-FED Stage 3 进行公平对比？**

## 📋 训练策略对比

### MD-FED Stage 3 的训练流程

```
Stage 1 (大数据集预训练)
    ↓
Stage 2 (目标数据集训练)
    ↓
Stage 3 (Few-shot Fine-tuning)
    ↓
在 manual_annotations.json 上 fine-tune
    ↓
得到最终模型
```

**关键点**：
- ✅ Stage 2 模型在相似任务上已经预训练
- ✅ Stage 3 只需要在小数据集 (85 个视频) 上 fine-tune
- ✅ 利用了大量预训练知识

### VTN 的训练流程

```
ImageNet 预训练 ViT (来自 timm)
    ↓
VTN = ViT (frozen/unfrozen) + Temporal Transformer
    ↓
在 manual_annotations.json 上训练
    ↓
得到最终模型
```

**关键点**：
- ✅ **Spatial backbone (ViT)**: 使用 ImageNet 预训练权重
- ✅ **Temporal transformer**: 从随机初始化开始
- ✅ 整个模型在 manual_annotations.json 上端到端训练

## 🎯 推荐的对比方案

### 方案 1: 公平对比 (推荐) ⭐⭐⭐⭐⭐

**VTN 在 manual_annotations.json 上从头训练（使用 ImageNet 预训练 ViT）**

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_vs_stage3 \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

**对比的公平性**：

| 模型 | 预训练来源 | 训练数据 | 公平性 |
|------|-----------|---------|--------|
| **MD-FED Stage 3** | Stage 2 (网球数据) | manual_annotations.json | ✅ |
| **VTN** | ImageNet (通用视觉) | manual_annotations.json | ✅ |

**为什么公平？**
- ✅ 两者都使用预训练权重（MD-FED 用 Stage 2，VTN 用 ImageNet）
- ✅ 两者都在相同的标注数据上训练/fine-tune
- ✅ 两者都面临小数据集的挑战
- ✅ 对比的是**模型架构的优劣**（CNN+GRU vs ViT+Transformer）

### 方案 2: 不使用 Stage 2（不推荐）

如果想更严格的对比，可以让 MD-FED 也不使用 Stage 2：

```bash
# MD-FED 从 Stage 1 直接到 Stage 3
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage1 \  # 用 stage1 而非 stage2
    --manual_annotations manual_annotations.json

# VTN 
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json
```

**但这不推荐**，因为：
- ❌ Stage 2 是 MD-FED 流程的重要部分
- ❌ 会严重损害 MD-FED 的性能
- ❌ 不能反映两个方法的真实能力

## 🔧 VTN 的训练细节

### 1. 预训练权重的使用

```python
# MD-FED/model/vtn.py, line 43-49
self.spatial_transformer = timm.create_model(
    f'vit_{spatial_size}_patch{patch_size}_{img_size}', 
    pretrained=True,  # ← 使用 ImageNet 预训练
    img_size=model_img_size,
    in_chans=3
)
```

**说明**：
- ✅ `pretrained=True` 自动下载并加载 ImageNet 预训练的 ViT 权重
- ✅ 这些权重是在 ImageNet-1K (1.2M 图像) 上训练的
- ✅ 提供强大的视觉表征能力

### 2. 训练策略

**默认配置（推荐）**：

```python
# Spatial backbone: 解冻，允许微调
spatial_frozen = False  # ViT 可以被训练

# Temporal transformer: 从随机初始化
# 在训练中学习时序建模
```

**可选配置（如果过拟合严重）**：

```python
# 冻结 spatial backbone
spatial_frozen = True  # 只训练 temporal transformer
```

### 3. 完整训练命令

```bash
# 推荐配置：解冻 ViT，端到端训练
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_unfrozen \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --batch_size 4
```

## 📊 对比实验设计

### 实验设置

| 设置项 | MD-FED Stage 3 | VTN | 状态 |
|--------|----------------|-----|------|
| **训练数据** | manual_annotations.json | manual_annotations.json | ✅ 相同 |
| **数据增强** | 随机裁剪 224×224 | 随机裁剪 224×224 | ✅ 相同 |
| **评估策略** | 中心裁剪 224×224 | 中心裁剪 224×224 | ✅ 相同 |
| **预训练** | Stage 2 (网球) | ImageNet (通用) | ⚠️ 不同但合理 |
| **视频数量** | 85 个 | 85 个 | ✅ 相同 |

### 运行对比实验

```bash
# 1. MD-FED Stage 3 Baseline
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3_baseline

# 2. VTN 对比实验
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

### 评估结果

两个模型都会输出相同的指标：

```
F1 (LCL)      - 事件定位 F1
F1 (element)  - 元素级别 F1
F1 (event)    - 事件级别 F1
Edit Score    - 序列编辑距离
```

## 🎮 不同配置的探索

### 配置 1: Small ViT (推荐起点)

```bash
python train_vtn_comparison.py \
    --vtn_spatial_size small \  # 22M params
    --batch_size 4 \
    --num_epochs 50
```

**适合**：
- ✅ 平衡性能和速度
- ✅ GPU 内存 ≥ 12GB
- ✅ 作为 baseline 对比

### 配置 2: Tiny ViT (快速实验)

```bash
python train_vtn_comparison.py \
    --vtn_spatial_size tiny \  # 5.7M params
    --batch_size 6 \
    --num_epochs 30
```

**适合**：
- ✅ 快速验证可行性
- ✅ GPU 内存 < 12GB
- ✅ 快速迭代

### 配置 3: Base ViT (最强性能)

```bash
python train_vtn_comparison.py \
    --vtn_spatial_size base \  # 86M params
    --batch_size 2 \
    --num_epochs 100 \
    --learning_rate 0.00005
```

**适合**：
- ✅ 追求最佳性能
- ✅ GPU 内存 ≥ 24GB
- ✅ 有充足的训练时间

### 配置 4: 冻结 Spatial Backbone

如果过拟合严重，可以尝试冻结 ViT：

```python
# 需要修改代码，在 train_vtn_comparison.py 中添加参数
# 或直接修改 VTN 初始化：
self.vtn = VTN(
    frames=clip_len,
    num_classes=num_classes,
    spatial_frozen=True,  # ← 冻结 ViT
    ...
)
```

## 📈 预期结果分析

### MD-FED Stage 3 的优势

- ✅ 轻量级模型 (~10M params)
- ✅ 多模态融合 (RGB + Flow + Skeleton)
- ✅ 专门为动作检测设计
- ✅ Stage 2 预训练在网球数据上

### VTN 的优势

- ✅ 强大的 Transformer 架构
- ✅ 全局时空建模能力
- ✅ ImageNet 预训练的通用视觉特征
- ✅ 可扩展性好（可以用更大的 ViT）

### 可能的结果

| 场景 | 预期 | 原因 |
|------|------|------|
| **小数据集 (85 视频)** | MD-FED 可能更好 | 专门设计 + 网球预训练 |
| **增加数据量** | VTN 可能更好 | Transformer 需要更多数据 |
| **计算资源受限** | MD-FED 更好 | 参数更少，速度更快 |
| **泛化到新场景** | VTN 可能更好 | ImageNet 预训练更通用 |

## 🔍 训练监控

### 检查训练是否正常

```bash
# 查看训练日志
tail -f vtn_outputs/vtn_comparison/train.log

# 预期输出
Epoch 1/50: Loss=2.345, F1(LCL)=0.123
Epoch 2/50: Loss=2.012, F1(LCL)=0.234
...
```

### 过拟合检测

```
训练集 F1 持续上升，验证集 F1 下降 → 过拟合

解决方法：
1. 减小模型 (用 tiny 而非 small/base)
2. 冻结 spatial backbone
3. 增强数据增强
4. 减少训练 epochs
```

### 欠拟合检测

```
训练集和验证集 F1 都很低 → 欠拟合

解决方法：
1. 增加模型容量 (用 small/base 而非 tiny)
2. 增加训练 epochs
3. 调整学习率
4. 检查数据预处理
```

## ⚠️ 重要注意事项

### 1. 不要与"纯推理"对比

**错误做法** ❌：
```bash
# 下载预训练的 VTN 模型，直接在你的数据上推理
# 这是不公平的！因为该模型没有在你的数据上训练
```

**正确做法** ✅：
```bash
# 在你的 manual_annotations.json 上训练 VTN
python train_vtn_comparison.py --manual_annotations manual_annotations.json
```

### 2. ImageNet 预训练是标准做法

- ✅ 几乎所有视觉模型都使用 ImageNet 预训练
- ✅ MD-FED 的 RegNet 也是 ImageNet 预训练的
- ✅ 这是**标准且公平**的做法

### 3. 对比要点

**唯一的变量应该是模型架构**：
- MD-FED: CNN (RegNet) + TSM + GRU
- VTN: ViT + Longformer

其他条件应该**尽可能相同**：
- ✅ 相同的训练数据
- ✅ 相同的数据增强
- ✅ 相同的评估策略
- ✅ 类似的预训练策略（都用预训练权重）

## 🎯 总结

### 推荐的训练和对比方法

1. **VTN 训练**：在 manual_annotations.json 上训练（使用 ImageNet 预训练 ViT）
2. **MD-FED baseline**：使用现有的 Stage 3 结果
3. **公平对比**：比较两者在相同测试集上的指标

### 完整流程

```bash
# Step 1: 训练 VTN
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50

# Step 2: 对比结果
# VTN 的结果在: vtn_outputs/comparison/best_model_metrics.json
# MD-FED 的结果在: MD-FED/md_fed_outputs/stage3/evaluation_results.json

# Step 3: 生成对比报告
python compare_results.py \
    --vtn_results vtn_outputs/comparison/best_model_metrics.json \
    --mdfed_results MD-FED/md_fed_outputs/stage3/evaluation_results.json
```

**这是公平且合理的对比方法！** 🎯🎾
