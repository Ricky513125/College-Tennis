# 公平对比实验配置

## 🎯 数据增强一致性

为了确保 **VTN** 与你现有方法的**公平对比**，两者使用**完全相同的数据预处理策略**。

## 📐 裁剪策略

### 你的现有方法

```
训练时: 随机裁剪到 224×224 (数据增强)
评估时: 中心裁剪到 224×224 (确定性推理)
```

### VTN 对比实验（已配置一致）

```
训练时: 随机裁剪到 224×224 (is_eval=False)
评估时: 中心裁剪到 224×224 (is_eval=True)
```

## ✅ 代码实现

### 训练集 - 随机裁剪

```python
train_data = ActionSeqDataset(
    classes, train_json,
    frame_dir, clip_len, dataset_len,
    is_eval=False,  # ← 关键：使用随机裁剪
    crop_dim=224,
    stride=2
)
```

**效果**：
- 从 398×224 的帧中**随机**裁剪 224×224 区域
- 每个 epoch 看到不同的裁剪区域
- 提供数据增强，减少过拟合

### 验证集 - 中心裁剪

```python
val_data = ActionSeqDataset(
    classes, val_json,
    frame_dir, clip_len, dataset_len // 4,
    is_eval=True,  # ← 关键：使用中心裁剪
    crop_dim=224,
    stride=2
)
```

**效果**：
- 从 398×224 的帧中**固定**裁剪中心 224×224 区域
- 每次评估看到相同的区域
- 确定性推理，结果可复现

## 📊 裁剪可视化

### 训练时 - 随机裁剪

```
原始帧 (398×224):
┌─────────────────────────────────────┐
│  [crop1]                            │ 第1个batch
│           [crop2]                   │ 第2个batch
│                    [crop3]          │ 第3个batch
└─────────────────────────────────────┘

每个 224×224 的裁剪位置都不同
```

### 评估时 - 中心裁剪

```
原始帧 (398×224):
┌─────────────────────────────────────┐
│           [固定中心区域]             │ 始终相同
│              224×224                │
└─────────────────────────────────────┘

始终从中心裁剪 224×224
```

## 🔧 完整训练命令

### 使用正方形 224×224 (推荐用于公平对比)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/fair_comparison_224 \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.0001
```

**说明**：
- ✅ 训练时从 398×224 随机裁剪 224×224
- ✅ 评估时从 398×224 中心裁剪 224×224
- ✅ 与你现有方法完全一致

### 使用矩形 384×224 (如果想保留更多信息)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/rect_384x224 \
    --img_width 384 \
    --img_height 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

**说明**：
- 训练时从 398×224 resize 到 384×224 (无随机裁剪)
- 评估时同样 resize 到 384×224
- 保留更多信息，但数据增强较少

## 📈 两种配置的对比

| 配置 | 训练增强 | 信息损失 | 与现有方法一致性 | 推荐用于对比 |
|------|----------|----------|------------------|--------------|
| **224×224 正方形** | ⭐⭐⭐⭐⭐ | 43.7% 宽度 | ✅ 完全一致 | ⭐⭐⭐⭐⭐ |
| **384×224 矩形** | ⭐⭐ | 3.5% 宽度 | ⚠️ 不同策略 | ⭐⭐⭐ |

## 🎮 对比实验建议

### 方案 1: 公平对比 (推荐)

使用 224×224 正方形裁剪，与你的现有方法保持一致：

```bash
# 你的现有方法
python few_shot_learning_stage3.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3_baseline

# VTN 对比实验
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_comparison \
    --crop_dim 224  # ← 与 baseline 一致
```

**优势**：
- ✅ 完全公平的对比
- ✅ 唯一变量是模型架构 (MD-FED vs VTN)
- ✅ 结果直接可比

### 方案 2: 额外实验

如果想探索矩形输入的潜力：

```bash
# 实验1: 标准对比 (224×224)
python train_vtn_comparison.py \
    --save_dir ./vtn_outputs/exp1_square224 \
    --crop_dim 224

# 实验2: 矩形输入 (384×224)
python train_vtn_comparison.py \
    --save_dir ./vtn_outputs/exp2_rect384 \
    --img_width 384 \
    --img_height 224

# 对比结果
python compare_results.py \
    exp1_square224 \
    exp2_rect384 \
    baseline_stage3
```

## 🔍 验证裁剪策略

创建验证脚本确认裁剪行为：

```python
# verify_crop_strategy.py
import torch
from MD-FED.dataset.input_process import ActionSeqDataset
from util.dataset import load_classes

# 加载数据集
classes = load_classes('MD-FED/data/f3set-tennis-sub/elements.txt')

# 训练集 - 应该使用随机裁剪
train_data = ActionSeqDataset(
    classes, 'train.json',
    frame_dir='/path/to/frames',
    clip_len=96,
    dataset_len=100,
    is_eval=False,  # 随机裁剪
    crop_dim=224
)

# 验证集 - 应该使用中心裁剪
val_data = ActionSeqDataset(
    classes, 'val.json',
    frame_dir='/path/to/frames',
    clip_len=96,
    dataset_len=25,
    is_eval=True,  # 中心裁剪
    crop_dim=224
)

print("训练集裁剪策略:", train_data._frame_reader._crop_transform)
print("验证集裁剪策略:", val_data._frame_reader._crop_transform)

# 预期输出：
# 训练集裁剪策略: RandomCrop(size=(224, 224), padding=None)
# 验证集裁剪策略: CenterCrop(size=(224, 224))
```

## 📝 关键代码位置

### ActionSeqDataset 初始化

```python
# MD-FED/dataset/input_process.py, line 532-595

class ActionSeqDataset(Dataset):
    def __init__(
        self,
        classes,
        label_file,
        frame_dir,
        clip_len,
        dataset_len,
        is_eval=True,  # ← 关键参数
        crop_dim=None,
        ...
    ):
        # 根据 is_eval 选择裁剪策略
        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, same_transform, ...
        )
```

### 裁剪策略选择

```python
# MD-FED/dataset/input_process.py, line 417-436

def _get_img_transforms(is_eval, crop_dim, ...):
    crop_transform = None
    if crop_dim is not None:
        if is_eval:
            # 评估时：中心裁剪
            crop_transform = transforms.CenterCrop(crop_dim)
        else:
            # 训练时：随机裁剪
            crop_transform = transforms.RandomCrop(crop_dim)
    return crop_transform, img_transform
```

## ⚠️ 重要提醒

### 1. 确保配置一致

对比实验时，**必须**检查：

```bash
# 检查你的 baseline 配置
cat MD-FED/md_fed_outputs/stage3/config.json | grep crop_dim

# VTN 使用相同的 crop_dim
python train_vtn_comparison.py --crop_dim <相同值>
```

### 2. 数据增强的影响

| 策略 | 训练变化 | 泛化能力 | 过拟合风险 |
|------|----------|----------|------------|
| **随机裁剪** | 每个epoch不同 | ⭐⭐⭐⭐⭐ | 低 |
| **固定裁剪** | 每个epoch相同 | ⭐⭐⭐ | 高 |

随机裁剪提供隐式数据增强，对小数据集特别重要。

### 3. 矩形输入的权衡

如果使用矩形输入 (384×224)：

**优势**：
- ✅ 保留更多原始信息
- ✅ 更少的信息损失

**劣势**：
- ❌ 较少的数据增强（resize vs crop）
- ❌ 与 baseline 不一致，对比不公平
- ❌ 可能过拟合（小数据集）

## 🎯 推荐方案

### 对于公平对比

**使用 224×224 正方形，随机/中心裁剪**：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/vtn_vs_mdfed \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

### 对于最佳性能（不一定公平对比）

**使用 384×224 矩形，保留更多信息**：

```bash
python train_vtn_comparison.py \
    --img_width 384 \
    --img_height 224 \
    --save_dir ./vtn_outputs/vtn_rect_best
```

## 📊 预期结果报告

实验完成后，报告应包含：

```
实验配置:
- 模型: MD-FED Stage 3 vs VTN
- 输入尺寸: 224×224 (从 398×224 裁剪)
- 训练增强: 随机裁剪
- 评估策略: 中心裁剪
- 其他超参数: [保持一致]

结果:
                F1 (LCL)  F1 (element)  F1 (event)  Edit Score
MD-FED Stage 3   0.XXX      0.XXX        0.XXX       0.XXX
VTN              0.XXX      0.XXX        0.XXX       0.XXX

结论:
- 在相同的数据增强策略下...
- VTN 相比 MD-FED 的优势/劣势...
```

## ✅ 总结

**当前配置已确保公平对比**：

| 组件 | MD-FED Baseline | VTN 对比 | 状态 |
|------|----------------|----------|------|
| 训练裁剪 | 随机 224×224 | 随机 224×224 | ✅ 一致 |
| 评估裁剪 | 中心 224×224 | 中心 224×224 | ✅ 一致 |
| 数据增强 | RandomCrop | RandomCrop | ✅ 一致 |
| 推理策略 | CenterCrop | CenterCrop | ✅ 一致 |

**现在可以进行公平的对比实验了！** 🎯🎾
