# 评估 Annotations.json 效果指南

本指南说明如何评估 f3set 模型对 college videos 的标注结果，并与 Stage 3 的效果进行对比。

## 快速开始

### 基本用法

```bash
python evaluate_annotations.py \
    --pred_annotations annotations.json \
    --gt_annotations manual_annotations.json \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --output f3set_evaluation_results.json
```

### 与 Stage 3 对比

```bash
# 1. 评估 f3set 模型
python evaluate_annotations.py \
    --pred_annotations annotations.json \
    --gt_annotations manual_annotations.json \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --output f3set_evaluation_results.json

# 2. 评估 Stage 3 模型（如果还没有结果文件）
# 运行 Stage 3 测试后会生成结果，或者手动创建结果文件

# 3. 对比两个模型
python evaluate_annotations.py \
    --pred_annotations annotations.json \
    --gt_annotations manual_annotations.json \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --output f3set_evaluation_results.json \
    --compare_with stage3_evaluation_results.json
```

## 参数说明

- `--pred_annotations`: f3set 模型的标注结果文件（annotations.json）
- `--gt_annotations`: 手动验证的真实标签文件（manual_annotations.json）
- `--elements_file`: 类别定义文件（elements.txt）
- `--delta`: 时间容差（帧数），默认 10（与 Stage 3 评估一致）
- `--output`: 输出结果文件路径
- `--compare_with`: Stage 3 结果文件路径（用于对比）

## 输出指标

脚本会计算与 Stage 3 相同的指标：

1. **Mean F1 (LCL)**: 事件定位 F1 分数
2. **Mean F1 (event)**: 事件级别 F1 分数
3. **Mean F1 (element)**: 元素级别 F1 分数
4. **Edit score**: 编辑距离分数

## 输出文件

1. **结果文件** (`f3set_evaluation_results.json`):
   ```json
   {
     "f1_lcl": 0.433,
     "f1_event": 0.110,
     "f1_element": 0.180,
     "edit_score": 40.23,
     "num_videos": 50,
     "delta": 10
   }
   ```

2. **错误序列文件** (`f3set_evaluation_results_errors.txt`):
   - 记录预测错误的视频序列
   - 格式与 Stage 3 的 `error_sequences.txt` 相同

3. **对比文件** (`model_comparison.json`):
   - 如果使用 `--compare_with`，会生成对比结果

## 数据格式要求

### annotations.json 格式

```json
[
  {
    "video": "video_name/rally_001",
    "num_frames": 599,
    "fps": 60.0,
    "events": [
      {
        "frame": 39,
        "label": "far_serve_middle"
      },
      {
        "frame": 120,
        "label": "near_return_bh_gs"
      }
    ]
  }
]
```

### manual_annotations.json 格式

应该与 `annotations.json` 格式相同，作为 ground truth。

## 完整示例

### 步骤 1: 评估 f3set 模型

```bash
python evaluate_annotations.py \
    --pred_annotations /path/to/f3set_annotations.json \
    --gt_annotations manual_annotations.json \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --delta 10 \
    --output f3set_results.json
```

输出示例：
```
============================================================
Evaluating Annotations
============================================================

Loading predictions from: /path/to/f3set_annotations.json
Loading ground truth from: manual_annotations.json
Loading classes from: MD-FED/data/f3set-tennis-sub/elements.txt
Found 15 classes
Prediction videos: 50
Ground truth videos: 50
Common videos: 50

============================================================
Evaluation Results
============================================================

Mean F1 (LCL): 0.450000
Mean F1 (event): 0.120000
Mean F1 (element): 0.190000
Edit score: 42.500000

============================================================

Results saved to: f3set_results.json
Error sequences saved to: f3set_results_errors.txt
```

### 步骤 2: 评估 Stage 3 模型

如果还没有 Stage 3 的结果文件，可以：

**选项 A**: 从 Stage 3 训练输出中提取

```bash
# 查看 Stage 3 训练输出
cat ./MD-FED/md_fed_outputs/stage3_01281253/loss.json | grep "val_edit"
```

**选项 B**: 运行 Stage 3 测试脚本

```bash
python test_stage2_on_manual_data.py \
    --checkpoint_dir ./MD-FED/md_fed_outputs/stage3_01281253 \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --flow_dir /path/to/flow \
    --output_dir ./stage3_test_results
```

然后从输出中提取指标，创建 `stage3_results.json`:
```json
{
  "f1_lcl": 0.433,
  "f1_event": 0.110,
  "f1_element": 0.180,
  "edit_score": 40.23
}
```

### 步骤 3: 对比两个模型

```bash
python evaluate_annotations.py \
    --pred_annotations /path/to/f3set_annotations.json \
    --gt_annotations manual_annotations.json \
    --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
    --output f3set_results.json \
    --compare_with stage3_results.json
```

输出示例：
```
============================================================
Model Comparison
============================================================

Metric               F3Set          Stage 3        Difference      Winner    
---------------------------------------------------------------------------
F1 (LCL)             0.450000       0.433000       -0.017000       F3Set     
F1 (event)           0.120000       0.110000       -0.010000       F3Set     
F1 (element)         0.190000       0.180000       -0.010000       F3Set     
Edit score           42.500000      40.230000      -2.270000       F3Set     

============================================================

Comparison saved to: model_comparison.json
```

## 结果解读

### F3Set vs Stage 3

- **F1 (LCL)**: 事件定位能力
  - 如果 F3Set > Stage 3: F3Set 在事件定位上更好
  - 如果 Stage 3 > F3Set: Stage 3 在事件定位上更好

- **F1 (event)**: 完整事件序列匹配
  - 衡量模型预测完整事件序列的准确度

- **F1 (element)**: 细粒度动作分类
  - 衡量模型识别具体动作元素（如 serve, return, fh, bh）的准确度

- **Edit score**: 序列相似度
  - 分数越高越好（0-100）
  - 衡量预测序列和真实序列的整体相似度

## 常见问题

### Q1: 视频名称不匹配

**问题**: "No common videos found!"

**解决方案**:
1. 检查 `annotations.json` 和 `manual_annotations.json` 中的 `video` 字段是否一致
2. 确保视频名称格式相同（如 `video_name/rally_001`）

### Q2: 标签格式不匹配

**问题**: 某些标签无法解析

**解决方案**:
1. 检查标签字符串格式（如 `far_serve_middle`）
2. 确保标签中的类别名在 `elements.txt` 中定义
3. 脚本会自动过滤特殊标记（如 `-`, `in`, `out` 等）

### Q3: 如何调整时间容差？

**解决方案**:
```bash
# 使用更宽松的容差（±20 帧）
python evaluate_annotations.py \
    --pred_annotations annotations.json \
    --gt_annotations manual_annotations.json \
    --delta 20 \
    ...
```

### Q4: 如何只评估特定视频？

**解决方案**:
修改脚本或预处理 JSON 文件，只保留要评估的视频。

## 进阶使用

### 批量评估多个模型

```bash
#!/bin/bash

# 评估多个 f3set 模型版本
for version in v1 v2 v3; do
    python evaluate_annotations.py \
        --pred_annotations f3set_${version}_annotations.json \
        --gt_annotations manual_annotations.json \
        --elements_file MD-FED/data/f3set-tennis-sub/elements.txt \
        --output f3set_${version}_results.json
done

# 对比所有结果
python compare_all_results.py f3set_*_results.json stage3_results.json
```

### 可视化对比结果

可以编写脚本将对比结果可视化：

```python
import json
import matplotlib.pyplot as plt

with open('model_comparison.json', 'r') as f:
    comparison = json.load(f)

metrics = ['f1_lcl', 'f1_event', 'f1_element', 'edit_score']
f3set_scores = [comparison[m]['f3set'] for m in metrics]
stage3_scores = [comparison[m]['stage3'] for m in metrics]

# 绘制对比图
x = range(len(metrics))
plt.bar([i - 0.2 for i in x], f3set_scores, 0.4, label='F3Set')
plt.bar([i + 0.2 for i in x], stage3_scores, 0.4, label='Stage 3')
plt.xticks(x, metrics)
plt.legend()
plt.ylabel('Score')
plt.title('Model Comparison')
plt.savefig('model_comparison.png')
```

## 总结

使用 `evaluate_annotations.py` 可以：

1. ✅ 评估 f3set 模型的标注效果
2. ✅ 计算与 Stage 3 相同的指标
3. ✅ 对比两个模型的效果
4. ✅ 生成详细的错误分析

这样你就可以客观地比较 f3set 模型和 Stage 3 模型在 college videos 上的表现！
