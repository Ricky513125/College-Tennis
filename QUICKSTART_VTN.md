# VTN 快速开始指南

## ⚠️ 重要提示：公平对比需要 Stage 1 预训练

**本文档提供的是快速测试方法（跳过 Stage 1）。为了与 MD-FED 进行完全公平的对比，请参考：**

📚 **[README_VTN_Complete_Training.md](README_VTN_Complete_Training.md)** - 完整的 Stage 1 → Stage 3 训练流程

| 训练方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **完整流程 (Stage 1 → 3)** | ✅ 公平对比<br>✅ 更好性能 | ⏱️ 需要 F3Set 数据<br>⏱️ 训练时间长 (Stage 1: 8-12h) | 正式对比实验 |
| **快速测试 (仅 Stage 3)** | ⚡ 快速验证<br>⚡ 不需要额外数据 | ❌ 性能可能较差<br>❌ 不够公平 | 快速原型验证 |

---

## 🎯 目标 (快速测试)

在你的 manual_annotations.json 数据上训练 VTN，并与 MD-FED Stage 3 进行对比。

## ⚡ 一分钟理解

### VTN 训练策略

**VTN 需要训练，不是直接推理！**

- ✅ Spatial backbone (ViT): 使用 ImageNet 预训练（自动下载）
- ✅ Temporal transformer: 从随机初始化开始训练
- ✅ 整个模型: 在 manual_annotations.json 上端到端训练

### 与 MD-FED 的对比

| 模型 | 预训练来源 | 训练数据 | 公平性 |
|------|-----------|---------|--------|
| **MD-FED Stage 3** | Stage 2 (网球数据) | manual_annotations.json | ✅ |
| **VTN** | ImageNet (通用视觉) | manual_annotations.json | ✅ |

**这是公平的对比！** 两者都使用预训练 + 相同数据训练。

## 🚀 三步完成对比

### Step 1: 训练 VTN (必须)

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

**说明**：
- `crop_dim 224`: 与你现有方法一致（随机裁剪训练，中心裁剪评估）
- `vtn_spatial_size small`: 22M 参数，平衡性能和速度
- `num_epochs 50`: 通常 30-50 轮即可收敛

**训练时间**: 约 2-4 小时 (取决于 GPU)

### Step 2: 对比结果

```bash
python compare_results.py \
    --vtn_results ./vtn_outputs/comparison/best_model_metrics.json \
    --mdfed_results ./MD-FED/md_fed_outputs/stage3/evaluation_results.json \
    --output model_comparison.json
```

**输出示例**：

```
指标                 MD-FED Stage 3       VTN                  差异            胜者      
─────────────────────────────────────────────────────────────────────────────────────
F1 (LCL)             0.850000            0.823000            -0.027000       MD-FED    
F1 (element)         0.780000            0.795000            +0.015000       VTN       
F1 (event)           0.720000            0.710000            -0.010000       MD-FED    
Edit Score           0.650000            0.680000            +0.030000       VTN       
```

### Step 3: 分析结果

根据对比结果：

- 如果 VTN 更好 → 考虑使用 VTN
- 如果 MD-FED 更好 → 继续使用 MD-FED
- 如果相当 → 考虑其他因素（速度、内存、可解释性）

## 📊 不同配置的选择

### 配置 1: 快速验证 (推荐第一次尝试)

```bash
python train_vtn_comparison.py \
    --crop_dim 224 \
    --vtn_spatial_size tiny \  # 5.7M 参数，快
    --batch_size 6 \
    --num_epochs 20  # 快速收敛
```

**优点**: 快速看到结果 (~30分钟)
**缺点**: 性能可能不如 small/base

### 配置 2: 标准对比 (推荐)

```bash
python train_vtn_comparison.py \
    --crop_dim 224 \
    --vtn_spatial_size small \  # 22M 参数，平衡
    --batch_size 4 \
    --num_epochs 50
```

**优点**: 平衡性能和速度
**缺点**: 需要 2-4 小时训练

### 配置 3: 最佳性能 (如果有时间和GPU)

```bash
python train_vtn_comparison.py \
    --crop_dim 224 \
    --vtn_spatial_size base \  # 86M 参数，强
    --batch_size 2 \
    --num_epochs 100 \
    --learning_rate 0.00005
```

**优点**: 可能获得最佳性能
**缺点**: 需要 24GB GPU，训练 8+ 小时

## ⚙️ 完整参数说明

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \        # 标注文件 (必需)
    --frame_dir /path/to/frames \                        # 帧目录 (必需)
    --save_dir ./vtn_outputs/exp1 \                      # 输出目录
    --crop_dim 224 \                                     # 裁剪尺寸
    --vtn_spatial_size small \                           # ViT 大小
    --vtn_temporal_type longformer \                     # 时序建模
    --clip_len 96 \                                      # 片段长度
    --batch_size 4 \                                     # 批大小
    --num_epochs 50 \                                    # 训练轮数
    --learning_rate 0.0001 \                             # 学习率
    --train_ratio 0.8                                    # 训练集比例
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--crop_dim` | **224** | 与现有方法一致 |
| `--vtn_spatial_size` | **small** | tiny (快) / small (平衡) / base (强) |
| `--vtn_temporal_type` | **longformer** | 适合长序列 |
| `--batch_size` | **4** | small: 4, tiny: 6, base: 2 |
| `--num_epochs` | **50** | 通常 30-50 轮收敛 |

## 🔍 训练监控

### 查看训练进度

```bash
# 查看最新日志
tail -f vtn_outputs/comparison/train.log

# 预期输出
Epoch 1/50: train_loss=2.345, val_loss=2.123, F1(LCL)=0.123
Epoch 2/50: train_loss=2.012, val_loss=1.987, F1(LCL)=0.234
...
```

### 判断训练状态

**正常训练**：
```
train_loss 逐渐下降
val_loss 逐渐下降
F1 指标逐渐上升
```

**过拟合**：
```
train_loss 下降，val_loss 上升
→ 解决: 用更小的模型 (tiny)，或减少 epochs
```

**欠拟合**：
```
train_loss 和 val_loss 都很高
→ 解决: 用更大的模型 (base)，或增加 epochs
```

## 🐛 常见问题

### Q1: GPU 内存不足 (OOM)

```bash
# 减小 batch_size
python train_vtn_comparison.py --batch_size 2

# 或使用更小的模型
python train_vtn_comparison.py --vtn_spatial_size tiny
```

### Q2: 训练很慢

```bash
# 使用更小的模型
python train_vtn_comparison.py --vtn_spatial_size tiny

# 减少 epochs
python train_vtn_comparison.py --num_epochs 20

# 减少数据量 (用于快速测试)
python train_vtn_comparison.py --train_ratio 0.5
```

### Q3: 找不到 ImageNet 预训练权重

VTN 会自动从 `timm` 库下载 ImageNet 预训练权重。如果网络问题：

```bash
# 手动下载（在 Python 中）
import timm
model = timm.create_model('vit_small_patch16_224', pretrained=True)
# 权重会自动缓存到 ~/.cache/torch/hub/checkpoints/
```

### Q4: 如何验证配置正确？

```bash
# 验证裁剪策略
python verify_crop_strategy.py \
    --data_dir vtn_data \
    --frame_dir /path/to/frames

# 应该看到:
# 训练集: RandomCrop ✅
# 验证集: CenterCrop ✅
```

## 📚 相关文档

- 📘 [`README_VTN_Training_Strategy.md`](README_VTN_Training_Strategy.md) - **详细训练策略说明**
- 📘 [`README_Fair_Comparison.md`](README_Fair_Comparison.md) - 公平对比配置
- 📘 [`README_VTN_Comparison.md`](README_VTN_Comparison.md) - 完整 VTN 指南
- 📘 [`README_Rectangular_Input.md`](README_Rectangular_Input.md) - 矩形输入支持

## ✅ 检查清单

在开始训练前，确认：

- [ ] 有 manual_annotations.json 文件
- [ ] 有帧目录 (frame_dir)
- [ ] GPU 内存 ≥ 12GB (small) 或 ≥ 6GB (tiny)
- [ ] 已安装依赖: `pip install timm einops`
- [ ] 理解 VTN 需要训练（不是直接推理）

在对比结果前，确认：

- [ ] MD-FED Stage 3 已训练完成
- [ ] VTN 已训练完成
- [ ] 两者使用相同的 crop_dim (224)
- [ ] 两者使用相同的数据集

## 🎯 预期结果

基于小数据集 (85 视频)：

| 场景 | 预期结果 |
|------|----------|
| **VTN 胜出** | 如果 Transformer 架构能更好地建模时空关系 |
| **MD-FED 胜出** | 如果多模态融合 (RGB+Flow+Skeleton) 更重要 |
| **相当** | 两种方法各有优势，取决于具体任务 |

## 💡 总结

1. **VTN 需要训练** - 使用 ImageNet 预训练 ViT + manual_annotations.json 训练
2. **公平对比** - 与 MD-FED Stage 3 对比是公平的（都用预训练+相同数据）
3. **快速开始** - 先用 `tiny` 模型快速验证，再用 `small` 正式对比
4. **结果分析** - 使用 `compare_results.py` 自动对比指标

**现在开始训练吧！** 🚀🎾
