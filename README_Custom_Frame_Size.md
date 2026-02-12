# 自定义帧尺寸配置 (398×224)

## 📐 你的帧尺寸

```
原始帧尺寸: 398 × 224 (宽 × 高)
```

## 🎯 推荐的 crop_dim 配置

### 选项 1: crop_dim = 224 ✅ **推荐**

使用完整高度，从宽度随机裁剪：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

**优点**：
- ✅ 使用完整高度，不损失纵向信息
- ✅ 从398宽度中随机crop 224，有一定的数据增强效果
- ✅ 224是ViT的标准尺寸，模型预训练权重更匹配
- ✅ 裁剪区域: 从398中取224，还有174像素的裁剪空间

**计算**：
```
高度: 224 → 224 (完全使用)
宽度: 398 → 224 (随机裁剪，裁剪范围: 0-174像素偏移)
```

### 选项 2: crop_dim = 192

平衡性能和信息保留：

```bash
python train_vtn_comparison.py \
    --crop_dim 192 \
    --vtn_spatial_size small \
    --batch_size 6
```

**优点**：
- ✅ 能被patch_size(16)整除: 192÷16=12
- ✅ 更多的裁剪增强空间
- ✅ 稍微降低计算量

### 选项 3: crop_dim = 112

如果GPU内存有限或想要更快训练：

```bash
python train_vtn_comparison.py \
    --crop_dim 112 \
    --vtn_spatial_size tiny \
    --batch_size 8
```

## 📊 不同配置对比

| crop_dim | 从398×224裁剪 | 信息保留 | 计算量 | GPU内存 | 推荐度 |
|----------|---------------|----------|--------|---------|--------|
| **224** | 224×224 | ⭐⭐⭐⭐⭐ | 高 | ~12GB | ⭐⭐⭐⭐⭐ |
| **192** | 192×192 | ⭐⭐⭐⭐ | 中高 | ~10GB | ⭐⭐⭐⭐ |
| **160** | 160×160 | ⭐⭐⭐ | 中 | ~8GB | ⭐⭐⭐ |
| **112** | 112×112 | ⭐⭐ | 低 | ~6GB | ⭐⭐ |

## 🔧 数据处理流程

### 当前的处理流程

```python
原始帧: 398 × 224
    ↓
[Random Crop] crop_dim × crop_dim
    ↓
如果 crop_dim = 224:
  - 高度: 224 → 224 (完全使用)
  - 宽度: 398 → 随机选择224列
    ↓
[Resize] (如果需要)
    ↓
[Normalize]
    ↓
输入模型: 224 × 224 (或其他crop_dim)
```

### crop_dim=224 的裁剪示例

```
原始帧 (398×224):
┌────────────────────────────────────┐
│                                    │  224像素
│  [████████████████]                │  高度
│   ↑              ↑                 │
│   可裁剪区域(224×224)               │
│   宽度偏移: 0-174                  │
└────────────────────────────────────┘
        398像素宽度
```

## ⚙️ VTN 特殊要求

### Patch Size 约束

VTN 使用 Vision Transformer，需要 `crop_dim` 能被 `patch_size` 整除：

```python
patch_size = 16

# 有效的 crop_dim 值（能被16整除）：
✅ 224 = 16 × 14  # 14×14 patches
✅ 192 = 16 × 12  # 12×12 patches
✅ 160 = 16 × 10  # 10×10 patches
✅ 112 = 16 × 7   # 7×7 patches
❌ 150 = 16 × 9.375  # 不能整除
❌ 200 = 16 × 12.5   # 不能整除
```

## 🎮 具体配置建议

### 配置 1: 最佳性能（推荐）

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./vtn_outputs/crop224 \
    --crop_dim 224 \
    --patch_size 16 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.0001
```

**适用场景**：
- ✅ GPU内存 ≥ 12GB
- ✅ 想要最佳性能
- ✅ 能充分利用帧的高度信息

### 配置 2: 平衡模式

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./vtn_outputs/crop192 \
    --crop_dim 192 \
    --vtn_spatial_size small \
    --batch_size 6 \
    --num_epochs 50
```

**适用场景**：
- ✅ GPU内存 8-12GB
- ✅ 平衡性能和速度
- ✅ 想要更多数据增强

### 配置 3: 快速测试

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./vtn_outputs/crop112 \
    --crop_dim 112 \
    --vtn_spatial_size tiny \
    --batch_size 8 \
    --num_epochs 20
```

**适用场景**：
- ✅ GPU内存 < 8GB
- ✅ 快速原型验证
- ✅ 训练速度优先

## 📈 预期效果分析

### crop_dim=224 (推荐)

```
输入: 398×224 → Crop: 224×224

优势:
1. 充分利用纵向信息（网球比赛场地主要是纵向）
2. 横向裁剪提供数据增强
3. ViT标准尺寸，预训练权重效果好
4. 足够的上下文信息用于动作识别

劣势:
1. 计算量较大
2. GPU内存需求高
3. 小数据集可能过拟合（但你有85个视频应该够）
```

### crop_dim=192

```
输入: 398×224 → Resize: 398×192 → Crop: 192×192

优势:
1. 减少计算量
2. 更多裁剪增强空间
3. 适中的模型规模

劣势:
1. 损失部分纵向信息
2. 需要先resize
```

## 🔍 验证你的帧尺寸

创建一个验证脚本：

```python
import os
from PIL import Image

def check_frame_sizes(frame_dir, num_samples=10):
    """检查帧的实际尺寸"""
    sizes = {}
    count = 0
    
    for root, dirs, files in os.walk(frame_dir):
        for file in files:
            if file.endswith('.jpg') and count < num_samples:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                size = f"{img.width}×{img.height}"
                sizes[size] = sizes.get(size, 0) + 1
                count += 1
                print(f"✓ {img_path}: {img.width}×{img.height}")
    
    print(f"\n帧尺寸统计:")
    for size, count in sizes.items():
        print(f"  {size}: {count} 个样本")
    
    return sizes

# 使用
check_frame_sizes("/path/to/frames")
```

预期输出：
```
✓ .../rally_xxx/000001.jpg: 398×224
✓ .../rally_xxx/000002.jpg: 398×224
...

帧尺寸统计:
  398×224: 10 个样本
```

## ⚠️ 重要提醒

### 1. 确保 MD-FED Stage 3 使用相同配置

如果你要对比 VTN 和 MD-FED Stage 3，**两者必须使用相同的 crop_dim**：

```bash
# 先查看 Stage 3 用的 crop_dim
python -c "
import json
with open('path/to/stage3/config.json') as f:
    print(f\"Stage 3 crop_dim: {json.load(f).get('crop_dim')}\")
"

# VTN 使用相同的值
python train_vtn_comparison.py --crop_dim <相同值>
```

### 2. 398×224 是特殊尺寸

这个尺寸比较少见。常见的分辨率：
- 1920×1080 (Full HD)
- 1280×720 (HD)
- 640×480 (VGA)

398×224 可能是：
- ✓ 预处理后的尺寸（已经裁剪过）
- ✓ 降采样的结果
- ✓ 特定比例的resize

确认这是预期的尺寸很重要！

## 🎯 最终推荐

基于你的 398×224 帧尺寸，**强烈推荐使用 crop_dim=224**：

```bash
# 完整命令
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/crop224_small \
    --crop_dim 224 \
    --patch_size 16 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --temporal_arch gru \
    --clip_len 96 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --num_epochs 50 \
    --train_ratio 0.8
```

这样可以：
- ✅ 完全利用224的高度
- ✅ 从398宽度中裁剪224，有数据增强效果
- ✅ 使用ViT标准尺寸
- ✅ 与原始帧尺寸最匹配

---

**总结**: 对于 398×224 的帧，推荐 `crop_dim=224` 以充分利用纵向信息！🎯
