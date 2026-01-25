# 训练问题分析与解决方案

## 问题描述

训练过程中出现以下问题：
1. **所有 F1 分数为 0.0** (LCL, event, element, Edit score)
2. **验证损失高于训练损失** (0.78 vs 0.77)
3. **损失值在 0.77-0.78 之间徘徊，没有明显下降**

## 根本原因分析

### 1. F1 分数为 0 的原因

查看评估代码 (`MD-FED/train_MD-FED.py` 第 445 行)：
```python
coarse_pred = np.argmax(coarse_scores, axis=1)  # 二分类：0=无事件, 1=有事件
```

如果模型始终预测类别 0（无事件），则：
- `coarse_pred` 全为 0
- `fine_pred = coarse_pred[:, np.newaxis] * fine_pred` (第 481 行) 会将所有细粒度预测置零
- 导致所有 F1 分数为 0

### 2. 验证损失高于训练损失

这可能是正常的，但结合 F1=0，更可能表明：
- 模型没有学习到有效特征
- 数据分布问题
- 模型架构或超参数设置不当

## 可能的原因

### 原因 1: 模型未学习
- 损失值在 0.77-0.78 之间徘徊，没有明显下降趋势
- 模型可能陷入了局部最优或梯度消失

### 原因 2: 数据不平衡
- 如果数据中"无事件"帧远多于"有事件"帧
- 模型可能学习到总是预测"无事件"的策略

### 原因 3: 学习率问题
- 学习率 0.001 可能过大或过小
- 需要根据实际训练情况调整

### 原因 4: Stage 1 特定问题
- Stage 1 只使用骨架数据 (skeleton)
- 如果骨架数据质量不好或预处理有问题，模型无法学习

## 解决方案

### 方案 1: 检查数据质量

运行诊断脚本：
```bash
python diagnose_training.py md_fed_outputs/stage1
```

检查：
1. 数据是否正确加载
2. 标签分布是否平衡
3. 骨架数据文件是否存在且有效

### 方案 2: 调整学习率

尝试不同的学习率：
```bash
# 降低学习率
python run_md_fed_stage1.py \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1_lr0001 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001  # 降低 10 倍
```

或者使用学习率调度器（代码中应该已经包含）。

### 方案 3: 检查模型输出

在评估函数中添加调试信息，查看：
1. `coarse_scores` 的分布
2. 是否所有预测都是类别 0
3. 模型输出的概率值

### 方案 4: 检查损失函数

确认：
1. 损失值是否正常（没有 NaN 或 Inf）
2. 梯度是否正常更新
3. 模型参数是否在变化

### 方案 5: 数据增强或采样策略

如果数据不平衡：
1. 使用加权损失函数
2. 对少数类进行过采样
3. 调整 batch 采样策略

## 立即检查项

1. **检查 loss.json**:
   ```bash
   cat md_fed_outputs/stage1/loss.json | python -m json.tool | head -20
   ```
   查看损失是否真的在下降

2. **检查 error_sequences.txt**:
   ```bash
   cat error_sequences.txt | head -50
   ```
   如果文件为空，说明模型没有预测任何事件

3. **检查数据准备**:
   ```bash
   python -c "
   import json
   with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:
       data = json.load(f)
   print(f'Total videos: {len(data)}')
   # 检查标签分布
   "
   ```

4. **检查骨架数据**:
   ```bash
   ls -lh /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis/*.pkl | head -5
   ```
   确认文件存在且大小合理

## 建议的训练参数调整

如果当前设置不工作，尝试：

```bash
python run_md_fed_stage1.py \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1_v2 \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 0.0005
```

或者更保守的设置：
```bash
python run_md_fed_stage1.py \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1_v3 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
```

## 预期行为

正常训练应该看到：
1. **训练损失逐渐下降**（从初始值开始下降）
2. **验证损失也下降**（可能略高于训练损失）
3. **F1 分数逐渐提高**（从接近 0 开始，逐渐增加到 0.1, 0.2, ...）
4. **Edit score 逐渐提高**

如果训练了 30+ 个 epoch 后 F1 仍然是 0，说明模型没有学习到有效特征。

## 下一步

1. 运行 `diagnose_training.py` 获取详细诊断
2. 检查数据质量和标签分布
3. 尝试调整学习率
4. 如果问题持续，检查模型架构和数据加载代码
