# Stage 1 类别不平衡问题修复指南

## 问题诊断结果

根据诊断脚本的分析，发现了**严重的类别不平衡问题**：

- **事件帧比例**: 仅 0.84% (99.16% 是非事件帧)
- **不平衡比例**: **118.4:1** 
- **这是导致模型效果差的主要原因！**

即使当前代码中类别权重设置为 20.0，对于 118:1 的不平衡来说，**远远不够**。

## 解决方案

### 方案 1: 增加类别权重（最简单，立即生效）

当前权重是 20.0，需要增加到 **100.0** 才能有效处理 118:1 的不平衡。

#### 自动修复（推荐）

运行修复脚本：

```bash
python fix_stage1_class_imbalance.py
```

这会：
1. 自动将类别权重从 20.0 增加到 100.0
2. 创建备份文件
3. 生成改进的训练脚本

#### 手动修复

编辑 `MD-FED/train_MD-FED.py` 第 343 行：

```python
# 原来：
class_weights = torch.tensor([1.0, 20.0]).to(self._device)

# 修改为：
class_weights = torch.tensor([1.0, 100.0]).to(self._device)  # Increased for severe class imbalance (118:1)
```

### 方案 2: 使用改进的训练参数

运行改进的训练脚本：

```bash
# 1. 先运行修复脚本（如果还没运行）
python fix_stage1_class_imbalance.py

# 2. 编辑 run_stage1_improved.py，修改 --pose_dir 路径

# 3. 运行改进的训练
python run_stage1_improved.py
```

改进的训练参数：

```bash
python run_md_fed_stage1.py \
    --data_dir md_fed_data \
    --dataset f3set-tennis-sub \
    --pose_dir /path/to/skeleton/pkl/files \
    --output_dir ./md_fed_outputs/stage1_improved \
    --num_epochs 150 \          # 大幅增加（从 50 到 150）
    --learning_rate 0.0003 \    # 更小的学习率（从 0.001 到 0.0003）
    --clip_len 128 \            # 增加（从 96 到 128）
    --stride 1 \                # 减少（从 2 到 1）
    --warm_up_epochs 10 \       # 增加 warmup
    --acc_grad_iter 2           # 梯度累积
```

### 方案 3: 实现 Focal Loss（高级，可选）

Focal Loss 专门设计用于处理类别不平衡问题。需要修改损失函数：

```python
# 在 MD-FED/train_MD-FED.py 中添加 Focal Loss
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# 然后在 Stage 1 的损失计算中使用：
# coarse_loss = focal_loss(coarse_pred.reshape(-1, 2), coarse_label.flatten())
```

## 为什么需要这些改进？

### 1. 类别权重 100.0

- 当前不平衡比例是 118:1
- 类别权重 20.0 只能处理约 20:1 的不平衡
- 需要 100.0 才能有效处理 118:1 的不平衡

### 2. 增加训练轮数到 150

- 类别不平衡导致模型需要更多时间学习少数类
- 50 个 epoch 可能不够
- 150 个 epoch 可以让模型充分学习事件模式

### 3. 降低学习率到 0.0003

- 高类别权重会使梯度变大
- 需要更小的学习率来稳定训练
- 0.0003 是一个平衡点

### 4. 增加 clip_len 到 128

- 当前平均每个 rally 有 4.8 个事件
- 96 帧可能无法捕获完整的事件序列
- 128 帧可以更好地捕获事件序列

### 5. 减少 stride 到 1

- 事件帧密度很低（0.84%）
- stride=2 可能会跳过事件帧
- stride=1 可以捕获所有事件帧

## 预期改进效果

实施这些改进后，预期可以达到：

| 指标 | 当前 | 预期改进后 |
|------|------|-----------|
| Event F1 | 0.020 | **0.20-0.35** |
| Element F1 | 0.142 | **0.30-0.45** |
| Edit Score | 27.6 | **45-65** |
| LCL F1 | 0.462 | **0.55-0.70** |

## 训练监控

训练时注意观察：

1. **训练损失**: 应该持续下降
2. **验证损失**: 应该也在下降（避免过拟合）
3. **Event F1**: 应该逐步提升（这是关键指标）
4. **预测序列长度**: 应该逐渐增加（不再只预测第一个事件）

如果 Event F1 在训练过程中提升到 0.15 以上，说明改进有效。

## 如果效果仍不理想

如果实施上述改进后效果仍不理想，可以考虑：

1. **进一步增加类别权重**到 150.0 或 200.0
2. **实现 Focal Loss**（比加权交叉熵更有效）
3. **使用 Class-Balanced Sampling**（在数据加载时确保每个 batch 包含足够事件帧）
4. **数据增强**（对事件帧进行过采样）
5. **检查 skeleton 数据质量**（Stage 1 只使用 skeleton）

## 快速开始

```bash
# 1. 运行修复脚本
python fix_stage1_class_imbalance.py

# 2. 编辑并运行改进的训练脚本
# 编辑 run_stage1_improved.py 中的 --pose_dir
python run_stage1_improved.py

# 3. 监控训练过程
# 关注 Event F1 是否提升
```

## 注意事项

1. **备份原始文件**: 修复脚本会自动创建备份
2. **逐步调整**: 如果 100.0 权重太大导致训练不稳定，可以尝试 50.0 或 75.0
3. **耐心等待**: 150 个 epoch 的训练需要较长时间
4. **定期检查**: 每 10-20 个 epoch 检查一次验证指标
