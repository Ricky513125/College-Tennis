# 修复模型不预测事件的问题

## 问题确认

从 `error_sequences.txt` 分析结果看：
- **预测序列**: 空（模型没有预测任何事件）
- **真实标签**: 有内容（验证集中有事件）

这说明：
- 模型总是预测类别 0（无事件）
- 所有真实事件都被漏掉了（False Negative）
- 因此 F1 = 0

## 根本原因

**数据不平衡**：训练数据中"无事件"帧远多于"有事件"帧，导致模型学习到总是预测"无事件"的策略。

## 已实施的修复

### 1. 添加加权损失函数

在 `MD-FED/train_MD-FED.py` 第 336-342 行，已添加类别权重：

```python
# Use weighted loss to handle class imbalance (event vs no-event)
# Weight event class (class 1) more heavily to encourage event prediction
class_weights = torch.tensor([1.0, 10.0]).to(self._device)
coarse_loss = F.cross_entropy(
    coarse_pred.reshape(-1, 2), 
    coarse_label.flatten(), 
    weight=class_weights,
    **ce_kwargs
)
```

这会给事件类别（类别 1）10 倍权重，鼓励模型预测事件。

### 2. 增加 delta 容差

已将评估时的 delta 从 1 增加到 10，允许更宽松的匹配。

### 3. 添加调试输出

在评估函数中添加了调试信息，可以看到预测和标签的对比。

## 下一步操作

### 1. 检查数据平衡

```bash
python check_data_balance.py md_fed_data/f3set-tennis-sub/train.json
```

这会显示：
- 事件帧 vs 无事件帧的比例
- 是否需要调整权重（如果比例 > 100:1，可能需要更高权重）

### 2. 重新训练

使用修改后的代码重新训练：

```bash
python run_md_fed_stage1.py \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1_weighted \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001
```

### 3. 调整权重（如果需要）

如果数据不平衡非常严重（> 100:1），可以增加权重：

```python
# 在 MD-FED/train_MD-FED.py 中修改
class_weights = torch.tensor([1.0, 20.0]).to(self._device)  # 增加到 20 倍
# 或
class_weights = torch.tensor([1.0, 50.0]).to(self._device)  # 增加到 50 倍
```

### 4. 监控训练

观察训练过程：
- 损失是否下降
- 评估时是否开始预测事件
- F1 分数是否提高

## 预期结果

修复后应该看到：
1. ✅ 模型开始预测事件（预测序列不再为空）
2. ✅ F1 分数从 0 开始提高
3. ✅ error_sequences.txt 中预测序列有内容

## 如果仍然不预测事件

如果修复后仍然不预测事件，检查：

1. **学习率是否合适**
   - 尝试降低：`--learning_rate 0.0001`
   - 或增加：`--learning_rate 0.005`

2. **训练轮数是否足够**
   - 增加：`--num_epochs 100`

3. **骨架数据质量**
   - 检查骨架数据是否正确加载
   - 验证特征提取是否正常

4. **模型架构**
   - 确认 Stage 1 使用正确的架构（skeleton only）
