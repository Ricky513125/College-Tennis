# 继续训练快速指南

## 问题诊断

你的训练结果显示最佳 epoch 是 80，Edit score 只有 21.72，而之前 500 epoch 训练的最佳 epoch 是 490，Edit score 是 49.21。这说明脚本可能没有正确加载之前的模型。

## 解决方案

### 步骤 1: 检查当前训练状态

```bash
python check_training_status.py ./MD-FED/md_fed_outputs/stage3
```

这会显示：
- 训练历史（总 epoch 数、最佳 epoch、最佳分数）
- 可用的检查点文件
- 最佳模型的完整路径

### 步骤 2: 从最佳 Epoch (490) 继续训练

**重要**: 必须使用 `--resume_from_best` 参数！

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

## 脚本改进

改进后的脚本会：

1. **自动检测已有训练**
   - 显示训练历史（总 epoch、最佳 epoch、最佳分数）
   - 列出所有可用的检查点

2. **清楚显示加载的模型**
   ```
   📂 Loading BEST checkpoint from:
      Path: /full/path/to/checkpoint_490.pt
      Epoch: 490
      Edit score: 49.2100
   ✓ Model loaded successfully
      Will continue from epoch 491
   ```

3. **显示训练配置**
   ```
   Training Configuration
   ============================================================
     Start epoch: 491
     Additional epochs: 200
     Total epochs: 691
     Previous best: epoch 490 (Edit score: 49.2100)
   ============================================================
   ```

4. **训练结束时显示最佳模型路径**
   ```
   📂 Best Model Checkpoint:
      Path: /full/path/to/checkpoint_XXX.pt
      Epoch: XXX
      Edit score: XX.XXXX
      Size: XX.XX MB
      ✓ Checkpoint file exists
   ```

## 关键参数说明

- `--resume_from_best`: **必须使用**，从最佳 epoch (490) 继续
- `--resume`: 从最后 epoch 继续（可能不是最佳的）
- `--resume_from_epoch N`: 从指定 epoch 继续
- `--learning_rate 0.00001`: 使用更小的学习率（原来的 1/10）

## 验证是否正确加载

运行脚本时，你应该看到类似这样的输出：

```
============================================================
Found existing training in: ./MD-FED/md_fed_outputs/stage3
============================================================

📊 Training History:
   Last epoch: 500
   Best epoch: 490
   Best Edit score: 49.2100
   Total epochs trained: 500

📁 Available checkpoints:
   Epochs: 0 to 500 (total: 501)
   ✓ Best checkpoint exists: ./MD-FED/md_fed_outputs/stage3/checkpoint_490.pt

🔄 Will resume from BEST epoch: 490 (Edit score: 49.2100)

============================================================
Loading Model
============================================================
📂 Loading BEST checkpoint from:
   Path: /full/path/to/checkpoint_490.pt
   Epoch: 490
   Edit score: 49.2100
✓ Model loaded successfully
   Will continue from epoch 491
```

如果看到 "Loading from Stage 2" 而不是 "Loading BEST checkpoint"，说明没有正确加载之前的模型！

## 常见错误

### 错误 1: 没有使用 --resume_from_best

**症状**: 看到 "Loading from Stage 2"  
**解决**: 添加 `--resume_from_best` 参数

### 错误 2: 检查点文件不存在

**症状**: "Checkpoint not found"  
**解决**: 运行 `check_training_status.py` 检查可用的检查点

### 错误 3: 从错误的 epoch 开始

**症状**: 最佳 epoch 不是 490  
**解决**: 使用 `--resume_from_epoch 490` 明确指定

## 完整示例

```bash
# 1. 检查当前状态
python check_training_status.py ./MD-FED/md_fed_outputs/stage3

# 2. 从最佳 epoch 继续训练
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

# 3. 训练完成后，检查新的最佳模型
python check_training_status.py ./MD-FED/md_fed_outputs/stage3
```

## 使用最佳模型进行测试

训练完成后，使用最佳模型进行测试：

```bash
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage3 \
    --epoch 490 \  # 或新的最佳 epoch
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally
```
