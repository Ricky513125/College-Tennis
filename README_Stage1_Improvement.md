# Stage 1 训练结果分析与改进指南

## 当前结果分析

从你的训练结果来看：

```
Mean F1 (LCL): 0.462
Mean F1 (event): 0.020  ⚠️ 非常低
Mean F1 (element): 0.142
Edit score: 27.625
```

**主要问题**：
1. **Event F1 只有 0.020** - 说明模型几乎无法正确预测事件序列
2. **模型只预测第一个事件** - 从预测序列看，模型主要只预测了 `near_serve`，没有预测后续事件
3. **序列长度不匹配** - 预测序列很短，而真实序列包含多个事件

## 可能的原因

### 1. 类别不平衡问题
- 事件帧远少于非事件帧（可能只有 1-5%）
- 模型倾向于预测"无事件"类别

### 2. 训练不充分
- 可能训练轮数不够
- 学习率设置不当
- 损失函数权重需要调整

### 3. 数据质量问题
- Skeleton 数据质量可能有问题
- 事件标注可能不完整
- 数据量可能不足

### 4. 模型配置问题
- `clip_len` 可能太短，无法捕获完整事件序列
- `stride` 可能太大，丢失了事件帧
- 损失函数权重需要优化

## 诊断步骤

### 步骤 1: 运行诊断脚本

```bash
python analyze_stage1_results.py \
    --annotations ./manual_annotations.json \
    --output_dir ./md_fed_outputs/stage1
```

这个脚本会：
- 分析数据质量和分布
- 检查类别不平衡情况
- 分析训练历史
- 提供具体的改进建议

### 步骤 2: 检查数据质量

检查 skeleton 数据：
```bash
# 检查 skeleton 文件是否存在
ls /path/to/skeleton/pkl/files/*.pkl | wc -l

# 检查一个 skeleton 文件的内容
python -c "
import pandas as pd
import sys
pkl = pd.read_pickle('path/to/skeleton.pkl')
print('Keys:', pkl.keys())
print('Keypoint shape:', pkl['keypoint'].shape if 'keypoint' in pkl else 'N/A')
"
```

## 改进方案

### 方案 1: 调整训练参数（推荐先试这个）

```bash
python run_md_fed_stage1.py \
    --data_dir md_fed_data \
    --dataset f3set-tennis-sub \
    --pose_dir /path/to/skeleton/pkl/files \
    --output_dir ./md_fed_outputs/stage1_v2 \
    --num_epochs 100 \          # 增加训练轮数
    --batch_size 4 \
    --learning_rate 0.0005 \    # 降低学习率（原来的一半）
    --clip_len 128 \            # 增加 clip 长度（从 96 增加到 128）
    --stride 1 \                # 减少 stride（从 2 减少到 1，增加采样密度）
    --warm_up_epochs 5           # 增加 warmup
```

### 方案 2: 使用 Focal Loss（处理类别不平衡）

需要修改 `MD-FED/train_MD-FED.py` 中的损失函数。可以创建一个补丁文件：

```python
# 在 Stage 1 的损失计算部分，使用 Focal Loss
# Focal Loss 公式: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```

### 方案 3: 增加类别权重

当前代码中类别权重是 20.0，可以尝试增加到 30.0-50.0：

修改 `MD-FED/train_MD-FED.py` 第 343 行：
```python
class_weights = torch.tensor([1.0, 50.0]).to(self._device)  # 从 20.0 增加到 50.0
```

### 方案 4: 数据增强和采样策略

1. **Class-balanced sampling**: 确保每个 batch 中包含足够的事件帧
2. **Oversampling**: 对包含事件的 clip 进行过采样
3. **数据增强**: 对 skeleton 数据进行轻微扰动

### 方案 5: 检查并修复数据

1. **检查 skeleton 数据质量**:
   ```python
   # 检查 skeleton 数据是否完整
   import pandas as pd
   pkl = pd.read_pickle('skeleton_file.pkl')
   print('Keypoint shape:', pkl['keypoint'].shape)
   print('Total frames:', pkl.get('total_frames', 'N/A'))
   ```

2. **检查标注完整性**:
   - 确保所有事件都被正确标注
   - 检查事件帧号是否正确

3. **验证数据对齐**:
   - 确保 JSON 中的 `video` 字段与 skeleton pkl 文件名匹配
   - 确保帧数一致

## 具体改进步骤（按优先级）

### 优先级 1: 快速改进（立即尝试）

1. **增加训练轮数到 100**
2. **降低学习率到 0.0005**
3. **增加 clip_len 到 128**
4. **减少 stride 到 1**

```bash
python run_md_fed_stage1.py \
    --data_dir md_fed_data \
    --dataset f3set-tennis-sub \
    --pose_dir /path/to/skeleton/pkl/files \
    --output_dir ./md_fed_outputs/stage1_improved \
    --num_epochs 100 \
    --learning_rate 0.0005 \
    --clip_len 128 \
    --stride 1
```

### 优先级 2: 处理类别不平衡

1. **增加类别权重到 50.0**
2. **使用学习率调度器**

### 优先级 3: 数据质量检查

1. 运行诊断脚本
2. 检查 skeleton 数据
3. 验证标注完整性

### 优先级 4: 高级优化

1. 实现 Focal Loss
2. 使用 class-balanced sampling
3. 数据增强

## 预期改进

实施这些改进后，预期可以达到：

- **Event F1**: 从 0.020 提升到 0.15-0.30
- **Element F1**: 从 0.142 提升到 0.25-0.40
- **Edit Score**: 从 27.6 提升到 40-60

## 监控训练过程

训练时注意观察：

1. **训练损失是否持续下降**
2. **验证损失是否也在下降**（避免过拟合）
3. **Event F1 是否在提升**
4. **预测序列长度是否在增加**

如果训练损失下降但验证损失不降或上升，说明过拟合，需要：
- 增加正则化
- 减少模型复杂度
- 增加数据量

## 下一步

1. 先运行诊断脚本了解具体情况
2. 实施优先级 1 的改进
3. 如果效果仍不理想，继续实施优先级 2 和 3
4. 考虑是否需要更多数据或调整模型架构
