# 理解预测序列格式和 F1=0 的原因

## error_sequences.txt 格式说明

`error_sequences.txt` 的格式是：

```
视频名
预测序列 (用 -> 连接)
空行
真实标签序列 (用 -> 连接)
------------------------
```

### 示例解析

```
20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_100393_100484
far_serve

------------------------
```

这表示：
- **视频名**: `20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_100393_100484`
- **预测序列**: `far_serve` (模型预测了1个事件)
- **真实标签**: 空 (验证集中这个视频没有事件标签，或者标签加载失败)

### 更复杂的示例

```
20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_105517_105727
far_serve->near_return_bh_gs->far_stroke_bh_gs->near_stroke_fh_gs->far_stroke_fh_gs

------------------------
```

这表示：
- **预测序列**: 5个事件，按时间顺序用 `->` 连接
- **真实标签**: 空

## 关键发现

从你提供的 `error_sequences.txt` 内容看：

1. ✅ **模型在预测事件** - 预测序列有内容（如 `far_serve`）
2. ❌ **真实标签序列为空** - 所有示例的真实标签行都是空的

## 为什么真实标签为空？

### 可能原因 1: 验证集中没有这些视频的标签

代码逻辑（`MD-FED/train_MD-FED.py` 第 528-534 行）：
```python
print_gts = []
for i in range(len(fine_pred)):
    if coarse_label[i] == 1:  # 只有当 coarse_label 为 1 时才添加
        print_gt = []
        for j in range(len(fine_pred[0])):
            if fine_label[i, j] == 1:
                print_gt.append(classes_inv[j + 1])
        print_gts.append('_'.join(print_gt))
```

如果 `coarse_label` 全为 0（没有事件），`print_gts` 就会是空的。

### 可能原因 2: 视频名不匹配

验证集使用的视频名格式可能与预测时使用的不同，导致 `dataset.get_labels(video)` 找不到对应的标签。

### 可能原因 3: 标签格式转换问题

标签从 JSON 文件加载后，需要转换成 `coarse_label` 和 `fine_label`。如果转换过程有问题，可能导致标签丢失。

## 如何检查

### 1. 检查验证集中这些视频是否有标签

```bash
python check_val_labels_for_videos.py md_fed_data/f3set-tennis-sub/val.json
```

### 2. 检查标签加载逻辑

查看 `MD-FED/dataset/input_process.py` 的 `get_labels` 方法：
- 是否正确从 JSON 读取 `events`
- `frame` 索引是否正确
- `label` 格式是否正确

### 3. 检查视频名匹配

```python
# 在评估函数中添加调试
print(f"Video in pred_dict: {video}")
print(f"Video in dataset.videos: {[v[0] for v in dataset.videos][:5]}")
```

## 关于序列格式的理解

你的理解是正确的：

1. **预测是序列格式** - 多个事件按时间顺序用 `->` 连接
2. **标签也需要序列格式** - 从验证集 JSON 文件中的 `events` 数组转换而来
3. **对比方式**:
   - **Edit Score**: 对比整个序列（使用编辑距离）
   - **F1 Score**: 对比每个事件的位置和类型（使用 delta 容差）

## 为什么 F1=0？

如果真实标签序列为空，意味着：
- `coarse_label` 全为 0
- 所有预测都是 False Positive（FP）
- True Positive (TP) = 0
- 因此 F1 = 0

## 解决方案

### 1. 检查验证集文件

确认 `md_fed_data/f3set-tennis-sub/val.json` 中这些视频是否有 `events`：

```python
import json
with open('md_fed_data/f3set-tennis-sub/val.json', 'r') as f:
    data = json.load(f)
    
# 查找特定视频
video_name = "20130607-M-Roland_Garros-SF-Novak_Djokovic-Rafael_Nadal_100393_100484"
for v in data:
    if video_name in v.get('video', ''):
        print(f"Found: {v['video']}")
        print(f"Events: {len(v.get('events', []))}")
        break
```

### 2. 检查视频名格式

验证集和预测使用的视频名格式可能不同，需要确保匹配。

### 3. 检查标签转换

确认 `get_labels` 方法正确地将 JSON 中的 `events` 转换为 `coarse_label` 和 `fine_label`。

## 下一步

运行以下命令来诊断：

```bash
# 1. 分析 error_sequences.txt 格式
python analyze_error_format.py MD-FED/error_sequences.txt

# 2. 检查验证集中的视频标签
python check_val_labels_for_videos.py md_fed_data/f3set-tennis-sub/val.json

# 3. 检查标签文件内容
python check_labels.py md_fed_data/f3set-tennis-sub/val.json
```
