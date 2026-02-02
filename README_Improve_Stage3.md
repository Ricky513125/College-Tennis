# 改进 Stage 3 训练效果指南

你已经完成了 500 个 epoch 的训练，最佳 epoch 是 490，Edit score 是 49.21。以下是继续提升效果的几种方法：

## 方法 1: 从最佳 Epoch 继续训练（推荐）

从最佳 epoch (490) 继续训练，使用更小的学习率进行精细微调：

```bash
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_best \
    --num_epochs 200 \
    --learning_rate 0.00001 \
    --eval_frequency 5 \
    --early_stop_patience 30
```

**关键参数**:
- `--resume_from_best`: 从最佳 epoch (490) 继续
- `--learning_rate 0.00001`: 使用更小的学习率（原来的 1/10）
- `--eval_frequency 5`: 每 5 个 epoch 评估一次（更频繁）
- `--early_stop_patience 30`: 30 个 epoch 无改进则停止

## 方法 2: 从最后 Epoch 继续训练

如果你想从最后一个 epoch (500) 继续：

```bash
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume \
    --num_epochs 200 \
    --reduce_lr \
    --eval_frequency 5
```

**关键参数**:
- `--resume`: 从最后 epoch 继续
- `--reduce_lr`: 自动将学习率降低 10 倍

## 方法 3: 从特定 Epoch 继续

如果你想从特定 epoch 继续（例如 epoch 450）：

```bash
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_epoch 450 \
    --num_epochs 200 \
    --learning_rate 0.00001
```

## 方法 4: 使用新的保存目录（重新开始但加载最佳模型）

如果你想在一个新的目录中继续训练，但使用之前的最佳模型：

```bash
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3_v2 \
    --num_epochs 200 \
    --learning_rate 0.00001
```

然后手动加载最佳模型：
```python
# 在脚本中，它会自动从 Stage 2 加载
# 但你可以修改脚本，让它从 stage3 的最佳 epoch 加载
```

## 优化建议

### 1. 学习率调整

**当前**: 0.0001  
**建议**: 
- 继续训练时使用 **0.00001** (1/10)
- 或者 **0.00005** (1/2)

```bash
--learning_rate 0.00001  # 精细微调
```

### 2. 更频繁的评估

**当前**: 只在最后 10 个 epoch 评估  
**建议**: 每 5-10 个 epoch 评估一次

```bash
--eval_frequency 5  # 每 5 个 epoch 评估
```

### 3. 早停策略

避免过拟合，如果长时间无改进则停止：

```bash
--early_stop_patience 30  # 30 个 epoch 无改进则停止
```

### 4. 调整 Batch Size

如果内存允许，可以增加 batch size：

```bash
--batch_size 8  # 从 4 增加到 8
```

### 5. 增加训练轮数

继续训练更多轮数，但使用更小的学习率：

```bash
--num_epochs 200  # 再训练 200 个 epoch
```

## 完整优化示例

```bash
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3_improved \
    --resume_from_best \
    --num_epochs 300 \
    --learning_rate 0.00001 \
    --batch_size 8 \
    --eval_frequency 5 \
    --early_stop_patience 50
```

## 监控训练过程

### 查看训练历史

```bash
# 查看 loss.json
cat ./MD-FED/md_fed_outputs/stage3/loss.json | tail -20

# 或者用 Python
python -c "
import json
with open('./MD-FED/md_fed_outputs/stage3/loss.json', 'r') as f:
    losses = json.load(f)
    print('Last 10 epochs:')
    for loss in losses[-10:]:
        print(f\"Epoch {loss['epoch']}: train={loss['train']:.5f}, val={loss['val']:.5f}, edit={loss.get('val_edit', 0):.4f}\")
"
```

### 找到最佳 epoch

```bash
python -c "
import json
with open('./MD-FED/md_fed_outputs/stage3/loss.json', 'r') as f:
    losses = json.load(f)
    best = max(losses, key=lambda x: x.get('val_edit', 0))
    print(f\"Best epoch: {best['epoch']}, Edit score: {best.get('val_edit', 0):.4f}\")
"
```

## 预期改进

使用这些优化方法，你可能看到：

1. **Edit score**: 49.21 → **52-55** (提升 5-10%)
2. **F1 (LCL)**: 0.433 → **0.45-0.48** (提升 4-11%)
3. **F1 (event)**: 0.110 → **0.13-0.15** (提升 18-36%)
4. **F1 (element)**: 0.180 → **0.20-0.22** (提升 11-22%)

## 其他改进方向

### 1. 增加训练数据

如果可能，增加更多手动验证的标注数据：

```bash
# 合并更多标注
cat manual_annotations.json additional_annotations.json > combined_annotations.json

# 使用合并后的数据重新训练
python improve_stage3_training.py \
    --manual_annotations combined_annotations.json \
    ...
```

### 2. 调整数据分割比例

尝试不同的训练/验证分割：

```bash
--train_ratio 0.9  # 使用 90% 数据训练，10% 验证
```

### 3. 使用不同的学习率调度

当前使用 Cosine Annealing，你也可以尝试：
- 固定学习率（更小的值）
- Step LR（在特定 epoch 降低学习率）

## 故障排除

### 问题 1: 找不到检查点

**错误**: `Checkpoint not found`

**解决**: 检查保存目录和 epoch 号：
```bash
ls -la ./MD-FED/md_fed_outputs/stage3/checkpoint_*.pt
```

### 问题 2: 学习率太大导致损失爆炸

**症状**: 损失突然变得很大或 NaN

**解决**: 使用更小的学习率：
```bash
--learning_rate 0.00001  # 或更小
```

### 问题 3: 内存不足

**解决**: 减小 batch size：
```bash
--batch_size 2  # 从 4 减小到 2
```

## 总结

**推荐方案**:

1. **从最佳 epoch 继续训练**（方法 1）
2. **使用更小的学习率** (0.00001)
3. **更频繁的评估** (每 5 个 epoch)
4. **早停策略** (30-50 个 epoch 无改进则停止)
5. **增加训练轮数** (200-300 个 epoch)

这样可以在不重新开始的情况下，进一步提升模型性能！
