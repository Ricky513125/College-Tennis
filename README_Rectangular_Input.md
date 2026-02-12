# 矩形输入支持 - 保留完整帧信息

## 🎯 核心优势

**不需要裁成正方形！** VTN 现在完全支持矩形输入，可以保留你的帧的完整信息。

## 📐 对于你的 398×224 帧

### 问题分析

你的原始帧尺寸：**398 × 224** (宽 × 高)

Vision Transformer 的要求：
- 图像的**高度**和**宽度**都必须能被 `patch_size` (16) 整除
- 398 ÷ 16 = **24.875** ❌ (不能整除)
- 224 ÷ 16 = **14** ✅ (能整除)

### ✅ 推荐方案：384×224

**最小损失的矩形输入**：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/rect_384x224 \
    --img_width 384 \
    --img_height 224 \
    --patch_size 16 \
    --vtn_spatial_size small \
    --num_epochs 50
```

**为什么选择 384×224？**

1. ✅ **384 ÷ 16 = 24** (能整除)
2. ✅ **224 ÷ 16 = 14** (能整除)
3. ✅ **只损失 14 像素宽度** (398 → 384)
4. ✅ **保留完整高度** (224 → 224)
5. ✅ **宽高比接近原始** (1.77 → 1.71)

### 📊 尺寸对比

| 配置 | 宽×高 | 损失宽度 | 损失高度 | Patches | 总Patches | 推荐度 |
|------|-------|----------|----------|---------|-----------|--------|
| **384×224** ⭐⭐⭐⭐⭐ | 384×224 | 14px (3.5%) | 0px | 24×14 | 336 | ⭐⭐⭐⭐⭐ |
| **400×224** | 400×224 | +2px (padding) | 0px | 25×14 | 350 | ⭐⭐⭐⭐ |
| **224×224** (正方形) | 224×224 | 174px (43.7%) | 0px | 14×14 | 196 | ⭐⭐⭐ |
| **112×112** (正方形) | 112×112 | 286px (71.9%) | 112px (50%) | 7×7 | 49 | ⭐⭐ |

## 🔧 详细配置

### 配置 1: 384×224 (推荐 - 最佳信息保留)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/rect_384x224_small \
    --img_width 384 \
    --img_height 224 \
    --patch_size 16 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.0001
```

**适用场景**：
- ✅ 想要最大化保留原始帧信息
- ✅ GPU 内存充足 (≥ 12GB)
- ✅ 网球场景（横向信息重要）

**特点**：
- 宽度: 398 → 384 (损失仅 3.5%)
- 高度: 224 → 224 (无损失)
- Patches: 24×14 = 336 个 16×16 的 patches
- 计算量: 比 224×224 高约 70%

### 配置 2: 400×224 (稍微 padding)

```bash
python train_vtn_comparison.py \
    --img_width 400 \
    --img_height 224 \
    --vtn_spatial_size small \
    --batch_size 4
```

**特点**：
- 宽度: 398 → 400 (需要 2px padding)
- 高度: 224 → 224 (无损失)
- Patches: 25×14 = 350 个
- 稍微多一点计算量，但信息更完整

### 配置 3: 224×224 (正方形 - 如果GPU受限)

```bash
python train_vtn_comparison.py \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --batch_size 6
```

**特点**：
- 从 398 宽度随机裁剪 224
- 损失约 43.7% 的横向信息
- 数据增强效果好
- 计算量适中

## 📈 数据处理流程

### 矩形输入处理 (384×224)

```python
原始帧: 398 × 224
    ↓
[Resize] 384 × 224
    ↓
切分为 patches (16×16):
  - 横向: 384 ÷ 16 = 24 个 patches
  - 纵向: 224 ÷ 16 = 14 个 patches
  - 总计: 24 × 14 = 336 个 patches
    ↓
[Flatten & Project]
    ↓
336 个 patch embeddings
    ↓
[Vision Transformer]
    ↓
输出特征: [batch, clip_len, vit_dim]
```

### 正方形输入处理 (224×224)

```python
原始帧: 398 × 224
    ↓
[Random Crop] 224 × 224
    ↓
切分为 patches:
  - 14 × 14 = 196 个 patches
    ↓
[Vision Transformer]
    ↓
输出特征
```

**对比**：
- 矩形保留更多信息 (336 vs 196 patches)
- 矩形无随机裁剪，更稳定
- 正方形有数据增强，可能泛化更好

## ⚙️ 实现细节

### 代码中的关键修改

#### 1. 模型初始化支持矩形

```python
model = VTN_MD_FED(
    num_classes=len(classes),
    clip_len=96,
    img_height=224,      # 指定高度
    img_width=384,       # 指定宽度
    patch_size=16,
    spatial_size='small',
    temporal_type='longformer'
)
```

#### 2. Vision Transformer 支持矩形

```python
# timm 的 ViT 原生支持矩形输入
self.spatial_transformer = timm.create_model(
    'vit_small_patch16_224',
    pretrained=True,
    img_size=(224, 384),  # (height, width) tuple
    in_chans=3
)
```

#### 3. 数据加载自动适配

```python
# 数据加载器会自动 resize 到指定尺寸
train_data = ActionSeqDataset(
    ...,
    crop_dim=max(img_height, img_width)  # 使用最大维度
)
```

## 📊 性能对比预期

### 计算复杂度

| 配置 | Patches | Self-Attention | 相对计算量 | GPU 内存 |
|------|---------|----------------|------------|----------|
| **384×224** | 336 | O(336²) | 1.71x | ~14GB |
| **224×224** | 196 | O(196²) | 1.00x | ~12GB |
| **112×112** | 49 | O(49²) | 0.25x | ~6GB |

### 准确度预期

基于网球视频特性（横向运动为主）：

| 配置 | 信息保留 | 数据增强 | 预期 F1 |
|------|----------|----------|---------|
| **384×224** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 最高 |
| **400×224** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 最高 |
| **224×224** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 较高 |
| **112×112** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 中等 |

## 🔍 验证尺寸

创建脚本验证你的帧尺寸：

```python
# check_frame_size.py
import os
from PIL import Image
from collections import Counter

def analyze_frame_sizes(frame_dir):
    sizes = []
    
    for root, dirs, files in os.walk(frame_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                sizes.append((img.width, img.height))
                if len(sizes) >= 100:  # 采样100张
                    break
        if len(sizes) >= 100:
            break
    
    size_counts = Counter(sizes)
    print("帧尺寸统计:")
    for size, count in size_counts.most_common():
        print(f"  {size[0]}×{size[1]}: {count} 张")
    
    # 推荐配置
    most_common_size = size_counts.most_common(1)[0][0]
    width, height = most_common_size
    
    print(f"\n检测到的主要尺寸: {width}×{height}")
    
    # 计算最佳矩形尺寸
    def round_to_multiple(n, base=16):
        return ((n + base - 1) // base) * base
    
    best_width = round_to_multiple(width - (width % 16))
    best_height = round_to_multiple(height - (height % 16))
    
    if best_width < width:
        # 向下取整到能被16整除
        best_width = (width // 16) * 16
    if best_height < height:
        best_height = (height // 16) * 16
    
    print(f"\n推荐配置:")
    print(f"  --img_width {best_width} \\")
    print(f"  --img_height {best_height}")
    print(f"\n损失:")
    print(f"  宽度: {width} → {best_width} (损失 {width - best_width}px, {(width - best_width) / width * 100:.1f}%)")
    print(f"  高度: {height} → {best_height} (损失 {height - best_height}px, {(height - best_height) / height * 100:.1f}%)")
    
    return best_width, best_height

# 使用
analyze_frame_sizes("/mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally")
```

预期输出：

```
帧尺寸统计:
  398×224: 100 张

检测到的主要尺寸: 398×224

推荐配置:
  --img_width 384 \
  --img_height 224

损失:
  宽度: 398 → 384 (损失 14px, 3.5%)
  高度: 224 → 224 (损失 0px, 0.0%)
```

## 🎮 完整训练命令

### 推荐配置 (384×224)

```bash
# 高质量模式 - Small ViT
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/rect_384x224_small \
    --img_width 384 \
    --img_height 224 \
    --patch_size 16 \
    --vtn_spatial_size small \
    --vtn_temporal_type longformer \
    --temporal_arch gru \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --train_ratio 0.8

# 快速模式 - Tiny ViT
python train_vtn_comparison.py \
    --img_width 384 \
    --img_height 224 \
    --vtn_spatial_size tiny \
    --batch_size 6 \
    --num_epochs 30
```

### 对比实验

```bash
# 实验1: 矩形 384×224
python train_vtn_comparison.py \
    --save_dir ./vtn_outputs/exp1_rect384x224 \
    --img_width 384 \
    --img_height 224

# 实验2: 正方形 224×224
python train_vtn_comparison.py \
    --save_dir ./vtn_outputs/exp2_square224 \
    --crop_dim 224

# 实验3: 正方形 112×112
python train_vtn_comparison.py \
    --save_dir ./vtn_outputs/exp3_square112 \
    --crop_dim 112 \
    --vtn_spatial_size tiny \
    --batch_size 8
```

## ⚠️ 重要提醒

### 1. Patch Size 约束

**必须满足**：
- `img_width % patch_size == 0`
- `img_height % patch_size == 0`

对于 patch_size=16：

```python
# 有效的宽度 (能被16整除)
✅ 384 = 16 × 24
✅ 400 = 16 × 25
✅ 416 = 16 × 26
❌ 398 = 16 × 24.875  # 不能整除！

# 有效的高度
✅ 224 = 16 × 14
✅ 240 = 16 × 15
✅ 256 = 16 × 16
```

### 2. 内存消耗

矩形输入通常需要更多GPU内存：

```python
内存需求 ≈ O(num_patches²) = O((H×W / patch_size²)²)

384×224: 336 patches → ~14GB
224×224: 196 patches → ~12GB
112×112: 49 patches → ~6GB
```

如果遇到 OOM (Out of Memory)：
- 减小 batch_size
- 使用更小的 spatial_size (tiny 而非 small/base)
- 考虑使用正方形输入

### 3. 与 MD-FED 对比要一致

如果要对比 VTN 和 MD-FED Stage 3，**确保使用相同的输入尺寸**：

```bash
# 检查 MD-FED Stage 3 的配置
cat ./MD-FED/md_fed_outputs/stage3/config.json | grep crop_dim

# VTN 使用相同配置
python train_vtn_comparison.py \
    --img_width <相同宽度> \
    --img_height <相同高度>
```

## 📝 命令行参数

```bash
# 矩形输入参数
--img_width INT          # 图像宽度（必须能被patch_size整除）
--img_height INT         # 图像高度（必须能被patch_size整除）

# 正方形输入参数（与矩形互斥）
--crop_dim INT          # 正方形尺寸（如果不指定img_width/height）

# 其他参数
--patch_size INT        # ViT patch大小（默认16）
--vtn_spatial_size STR  # ViT大小：tiny/small/base/large
```

## 🎯 总结

### 对于你的 398×224 帧

**最佳选择**: `--img_width 384 --img_height 224`

**优势**：
- ✅ 只损失 3.5% 宽度
- ✅ 保留 100% 高度
- ✅ 完整的场景信息
- ✅ 网球横向运动清晰可见
- ✅ 无随机裁剪，训练更稳定

**完整命令**：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/rect_384x224_best \
    --img_width 384 \
    --img_height 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

这样就能**保留几乎所有原始信息**，不需要裁成正方形！🎾✨
