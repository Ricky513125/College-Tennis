# Stage 3 损失计算详解

## 概述

Stage 3（少样本微调）使用**有监督学习**，损失函数由两部分组成：
1. **Coarse-grained Loss（粗粒度损失）**：事件定位损失
2. **Fine-grained Loss（细粒度损失）**：动作分类损失

## 损失函数公式

```
Total Loss = Coarse Loss + Fine Loss
```

---

## 1. Coarse-grained Loss（粗粒度损失）

### 目的
检测事件是否发生（二分类：事件 vs 非事件）

### 计算方式

```python
# 1. 模型输出
coarse_pred: [batch_size, clip_len, 2]  # 每个帧的2类logits（非事件/事件）

# 2. 真实标签
coarse_label: [batch_size, clip_len]  # 每个帧的标签（0=非事件, 1=事件）

# 3. 加权交叉熵损失（处理类别不平衡）
class_weights = [1.0, 50.0]  # 非事件权重1.0，事件权重50.0
coarse_loss = CrossEntropyLoss(
    coarse_pred.reshape(-1, 2),      # 展平为 [batch*clip_len, 2]
    coarse_label.flatten(),           # 展平为 [batch*clip_len]
    weight=class_weights              # 类别权重
)
```

### 关键点

1. **类别不平衡处理**
   - 事件帧通常只占 0.84% 的总帧数（严重不平衡，比例约 118:1）
   - 使用加权交叉熵：事件类别权重 50.0，非事件权重 1.0
   - 这样可以鼓励模型预测更多事件，避免模型总是预测"非事件"

2. **为什么权重是 50.0？**
   - 原本尝试过 100.0，但会导致数值不稳定（NaN）
   - 50.0 是平衡预测能力和数值稳定性的折中

3. **数学公式**
   ```
   L_coarse = -Σ w[y] * log(softmax(pred)[y])
   ```
   其中：
   - `w[0] = 1.0`（非事件权重）
   - `w[1] = 50.0`（事件权重）
   - `y` 是真实标签

---

## 2. Fine-grained Loss（细粒度损失）

### 目的
在检测到事件后，进一步分类具体的动作类型（多标签分类）

### 计算方式

```python
# 1. 模型输出
fine_pred: [batch_size, clip_len, num_classes]  # 每个帧的每个类别的logits

# 2. 真实标签
fine_label: [batch_size, clip_len, num_classes]  # 每个帧的每个类别的标签（0/1）

# 3. 掩码（只计算事件帧的损失）
coarse_mask: [batch_size, clip_len]  # 哪些帧是有效的事件帧
fine_mask = coarse_label.unsqueeze(2)  # 扩展为 [batch, clip_len, num_classes]

# 4. 二元交叉熵损失（多标签）
fine_bce_loss = BCEWithLogitsLoss(
    fine_pred, 
    fine_label.float(), 
    reduction='none'  # 不自动求和
)

# 5. 应用掩码（只计算事件帧的损失）
masked_fine_loss = fine_bce_loss * fine_mask

# 6. 归一化（除以事件帧总数）
fine_loss = masked_fine_loss.sum() / fine_mask.sum()
```

### 关键点

1. **多标签分类**
   - 每个事件帧可能有多个动作元素（如：`far` + `serve` + `middle`）
   - 使用二元交叉熵（BCE），每个类别独立预测

2. **掩码机制**
   - 只计算事件帧（`coarse_label == 1`）的损失
   - 非事件帧不参与细粒度损失计算
   - 这样可以避免模型在非事件帧上学习无意义的分类

3. **归一化**
   - 除以事件帧总数，确保损失不会因为事件帧数量变化而波动

4. **数学公式**
   ```
   L_fine = (1 / N_events) * Σ_{i in events} Σ_{j in classes} BCE(pred[i,j], label[i,j])
   ```
   其中：
   - `N_events` 是事件帧总数
   - `BCE` 是二元交叉熵

---

## 3. 总损失

```python
total_loss = coarse_loss + fine_loss
```

### 特点

- **简单相加**：两个损失直接相加，没有额外的权重平衡
- **梯度反向传播**：总损失用于反向传播，同时优化事件定位和动作分类

---

## 4. 损失计算流程图

```
输入数据
├── frame: RGB 帧 [B, L, C, H, W]
├── flow: 光流 [B, L, 2, H, W]
└── labels
    ├── coarse_label: 事件/非事件 [B, L]
    └── fine_label: 动作类别 [B, L, num_classes]

模型前向传播
└── Stage 3: model(frame, flow, None)
    ├── coarse_pred: [B, L, 2]
    └── fine_pred: [B, L, num_classes]

损失计算
├── Coarse Loss
│   ├── 加权交叉熵（权重 [1.0, 50.0]）
│   └── 处理类别不平衡
│
└── Fine Loss
    ├── 二元交叉熵（多标签）
    ├── 掩码（只计算事件帧）
    └── 归一化（除以事件帧数）

总损失 = Coarse Loss + Fine Loss
```

---

## 5. 代码实现细节

### 5.1 类别权重设置

```python
# 处理严重的类别不平衡
class_weights = torch.tensor([1.0, 50.0]).to(device)
# 非事件: 1.0
# 事件: 50.0（50倍权重，因为事件帧太少）
```

### 5.2 掩码应用

```python
# 粗粒度掩码：哪些帧是有效的事件帧
coarse_mask: [B, L]

# 细粒度掩码：扩展为 [B, L, num_classes]
fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
# 只有事件帧（coarse_label == 1）的损失会被计算
```

### 5.3 NaN 处理

```python
# 检查损失是否为 NaN
if math.isnan(coarse_loss.item()):
    print("Warning: Coarse loss is nan!")
    # 不添加到总损失中
else:
    loss += coarse_loss

if math.isnan(fine_loss.item()):
    print("Warning: Fine loss is nan!")
    # 不添加到总损失中
else:
    loss += fine_loss
```

---

## 6. 损失函数的选择原因

### 为什么使用加权交叉熵？

1. **类别不平衡严重**
   - 事件帧：0.84%
   - 非事件帧：99.16%
   - 如果不加权，模型会倾向于总是预测"非事件"

2. **权重选择**
   - 50.0 是经验值
   - 太小（如 10.0）：模型仍然预测不足
   - 太大（如 100.0）：数值不稳定，出现 NaN

### 为什么使用二元交叉熵（BCE）？

1. **多标签分类**
   - 每个事件可能有多个动作元素
   - 例如：`far_serve_middle` = `far` + `serve` + `middle`
   - BCE 允许每个类别独立预测

2. **掩码机制**
   - 只计算事件帧的损失
   - 避免在非事件帧上学习无意义的分类

---

## 7. 损失值示例

### 典型的损失值范围

- **Coarse Loss**: 0.1 - 2.0
  - 如果模型总是预测非事件，损失会很高（接近 2.0）
  - 如果模型能正确检测事件，损失会降低（接近 0.1）

- **Fine Loss**: 0.05 - 0.5
  - 如果动作分类准确，损失较低（0.05-0.1）
  - 如果分类错误，损失较高（0.3-0.5）

- **Total Loss**: 0.15 - 2.5
  - 训练初期：较高（1.0-2.5）
  - 训练后期：较低（0.15-0.5）

### 训练过程中的变化

```
Epoch 0:   Coarse=1.5, Fine=0.3, Total=1.8
Epoch 100: Coarse=0.8, Fine=0.2, Total=1.0
Epoch 200: Coarse=0.5, Fine=0.15, Total=0.65
Epoch 300: Coarse=0.3, Fine=0.1, Total=0.4
Epoch 490: Coarse=0.25, Fine=0.08, Total=0.33  # 最佳 epoch
```

---

## 8. 优化器设置

### 学习率

```python
# Stage 3 使用较小的学习率（微调）
learning_rate = 0.0001  # 默认值

# 学习率调度
- Warmup: 3 epochs（线性从 0.01 到 1.0）
- Cosine Annealing: 剩余 epochs（从 1.0 到 0）
```

### 梯度累积

```python
acc_grad_iter = 1  # 每个 batch 更新一次
# 如果 batch size 较小，可以增加梯度累积
```

---

## 9. 常见问题

### Q1: 为什么损失有时是 NaN？

**A**: 可能的原因：
1. 学习率太大
2. 类别权重太大（如 100.0）
3. 梯度爆炸

**解决方案**：
- 降低学习率
- 降低类别权重（如从 100.0 降到 50.0）
- 使用梯度裁剪

### Q2: 为什么 Coarse Loss 比 Fine Loss 大？

**A**: 这是正常的：
- Coarse Loss 计算所有帧（事件+非事件）
- Fine Loss 只计算事件帧（数量少）
- 由于类别不平衡，Coarse Loss 通常更大

### Q3: 如何调整损失权重？

**A**: 可以修改代码：

```python
# 当前：简单相加
total_loss = coarse_loss + fine_loss

# 可以尝试加权
total_loss = 0.7 * coarse_loss + 0.3 * fine_loss
# 或者
total_loss = coarse_loss + 0.5 * fine_loss
```

---

## 10. 总结

Stage 3 的损失计算：

1. **Coarse Loss**：加权交叉熵，处理类别不平衡
2. **Fine Loss**：掩码的二元交叉熵，多标签分类
3. **Total Loss**：两者相加，同时优化定位和分类

关键设计：
- ✅ 类别权重（50.0）处理不平衡
- ✅ 掩码机制只计算事件帧
- ✅ NaN 检测和处理
- ✅ 归一化确保损失稳定

这些设计使得模型能够在少样本场景下有效学习事件检测和动作分类。
