# 修复 F1 分数为 0 的问题

## 问题确认

根据诊断结果：
- ✅ **训练损失在下降** (0.842 → 0.771) - 模型在学习
- ⚠️ **验证损失高于训练损失** (37/50 epochs) - 可能过拟合
- 🔴 **所有 F1 分数为 0** - 模型没有预测任何事件
- ⚠️ **评估可能没有运行** - 所有 val_edit 为 0

## 根本原因

**模型始终预测类别 0（无事件）**，导致：
- `coarse_pred` 全为 0
- `fine_pred = coarse_pred * fine_pred` 全为 0
- 所有 F1 分数为 0

## 检查步骤

### 1. 检查 error_sequences.txt 位置

评估函数会在**当前工作目录**创建 `error_sequences.txt`。由于训练脚本会 `chdir` 到 `MD-FED/`，文件应该在：

```bash
# 检查多个可能的位置
ls -la error_sequences.txt
ls -la MD-FED/error_sequences.txt
ls -la MD-FED/md_fed_outputs/stage1/error_sequences.txt
find . -name "error_sequences.txt" 2>/dev/null
```

如果文件存在但为空，说明模型确实没有预测任何事件。

### 2. 检查数据平衡

检查训练数据中"事件"和"无事件"帧的比例：

```python
import json

# 检查训练数据
with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:
    train_data = json.load(f)

event_count = 0
no_event_count = 0

for video_data in train_data:
    for frame_data in video_data.get('frames', []):
        if frame_data.get('coarse_label', 0) == 1:
            event_count += 1
        else:
            no_event_count += 1

print(f"Event frames: {event_count}")
print(f"No-event frames: {no_event_count}")
print(f"Ratio: {event_count / (event_count + no_event_count) * 100:.2f}%")
```

如果事件帧比例 < 5%，数据严重不平衡，模型会倾向于预测"无事件"。

### 3. 检查评估是否运行

查看训练日志，寻找：
- `Mean F1 (LCL): 0.0` 这样的输出
- 如果有这些输出，说明评估运行了但返回 0
- 如果没有，说明评估没有运行

## 解决方案

### 方案 1: 使用加权损失函数（推荐）

如果数据不平衡，在损失函数中给事件类别更高权重：

修改 `MD-FED/train_MD-FED.py` 第 334 行附近：

```python
# 原来的代码
coarse_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs)

# 改为加权损失
class_weights = torch.tensor([1.0, 10.0]).to(self._device)  # 给事件类别 10 倍权重
coarse_loss = F.cross_entropy(
    coarse_pred.reshape(-1, 2), 
    coarse_label.flatten(), 
    weight=class_weights,
    **ce_kwargs
)
```

### 方案 2: 降低学习率

尝试更小的学习率：

```bash
python run_md_fed_stage1.py \
    --pose_dir /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1_v2 \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001  # 从 0.001 降低到 0.0001
```

### 方案 3: 检查数据准备

确认数据准备正确：

```bash
# 检查数据文件
ls -lh md_fed_data/f3set-tennis-sub/
# 应该看到: train.json, val.json, elements.txt

# 检查数据内容
python -c "
import json
with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:
    data = json.load(f)
print(f'Total videos: {len(data)}')
# 检查第一个视频的标签
if len(data) > 0:
    frames = data[0].get('frames', [])
    events = [f for f in frames if f.get('coarse_label') == 1]
    print(f'First video: {len(frames)} frames, {len(events)} events')
"
```

### 方案 4: 手动运行评估

创建一个脚本来手动运行评估并查看预测分布：

```python
# manual_eval.py
import sys
import os
sys.path.insert(0, 'MD-FED')

from train_MD-FED import MD_FED, evaluate
from util.dataset import load_classes
from dataset.input_process import ActionSeqVideoDataset
import torch

# 加载模型
checkpoint_path = 'MD-FED/md_fed_outputs/stage1/checkpoint_049.pt'
model = MD_FED(...)  # 使用相同的参数
model.load(torch.load(checkpoint_path))

# 加载验证数据
classes = load_classes('md_fed_data/f3set-tennis-sub/elements.txt')
val_data = ActionSeqVideoDataset(...)

# 运行评估
val_edit = evaluate(model, val_data, classes, window=5, dataset_name='f3set-tennis-sub')
print(f"Edit score: {val_edit}")
```

### 方案 5: 检查骨架数据

确认骨架数据正确加载：

```bash
# 检查骨架文件
ls -lh /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis/*.pkl | head -5

# 检查文件大小（应该不是 0）
find /home/lingyu/Tennis/data/TENNIS/skeletons/f3set-tennis -name "*.pkl" -size 0
```

## 立即行动

1. **运行更新的诊断脚本**：
   ```bash
   python diagnose_training.py MD-FED/md_fed_outputs/stage1
   ```

2. **检查 error_sequences.txt**：
   ```bash
   find . -name "error_sequences.txt" -exec cat {} \;
   ```

3. **检查数据平衡**：
   ```bash
   python -c "
   import json
   with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:
       data = json.load(f)
   events = sum(1 for v in data for f in v.get('frames', []) if f.get('coarse_label') == 1)
   total = sum(len(v.get('frames', [])) for v in data)
   print(f'Event frames: {events}/{total} ({events/total*100:.2f}%)')
   "
   ```

4. **如果数据不平衡，使用加权损失重新训练**

## 预期结果

修复后应该看到：
- F1 分数逐渐从 0 增加到 0.1, 0.2, ...
- Edit score 逐渐提高
- error_sequences.txt 包含预测错误（说明模型在预测，只是不够准确）
