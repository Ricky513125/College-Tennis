# 对比模型文档索引

所有对比实验相关的文档和工具汇总，包括 **VTN** 和 **I3D** 模型。

## 🚀 快速开始

### 新手必读

1. **[QUICKSTART_Comparison_Models.md](QUICKSTART_Comparison_Models.md)** ⭐⭐⭐⭐⭐
   - **最重要！先读这个！**
   - VTN、I3D、MD-FED 三模型对比
   - 3 步完成训练和评估

2. **[QUICKSTART_VTN.md](QUICKSTART_VTN.md)** ⭐⭐⭐⭐⭐
   - VTN 专用快速开始指南
   - 3 步完成 VTN 训练和对比
   - 包含所有常见问题解答

## 📚 模型文档

### VTN (Vision Transformer Network)

#### 核心概念

3. **[README_VTN_Training_Strategy.md](README_VTN_Training_Strategy.md)** ⭐⭐⭐⭐⭐
   - VTN 应该如何训练？
   - 为什么与 MD-FED Stage 3 的对比是公平的？
   - 详细的训练策略说明

4. **[README_VTN_Comparison.md](README_VTN_Comparison.md)** ⭐⭐⭐⭐
   - VTN 完整使用指南
   - 所有配置选项说明
   - 完整的对比流程

#### 高级主题

5. **[README_Rectangular_Input.md](README_Rectangular_Input.md)** ⭐⭐⭐
   - 支持矩形输入 (384×224)
   - 保留更多原始帧信息
   - 不一定需要裁成正方形

6. **[README_Crop_Dim_Explanation.md](README_Crop_Dim_Explanation.md)** ⭐⭐⭐
   - crop_dim 参数详解
   - 为什么选择 112 或 224
   - 对小数据集的影响

7. **[FIX_HTTP_403_ERROR.md](FIX_HTTP_403_ERROR.md)** ⭐⭐⭐
   - 解决预训练权重下载失败问题
   - 多种解决方案
   - 包含下载脚本

### I3D (Inflated 3D ConvNet)

8. **[README_I3D_Comparison.md](README_I3D_Comparison.md)** ⭐⭐⭐⭐
   - I3D 完整使用指南
   - 3D 卷积网络详解
   - 与 VTN 的对比

### 通用文档

9. **[README_Fair_Comparison.md](README_Fair_Comparison.md)** ⭐⭐⭐⭐
   - 数据增强策略一致性
   - 训练时随机裁剪，评估时中心裁剪
   - 确保公平对比的关键配置

10. **[README_Input_Data_Format.md](README_Input_Data_Format.md)** ⭐⭐
    - 输入数据格式说明
    - 模型期望的输入格式
    - 帧、光流、骨架数据

## 🛠️ 工具脚本

### 训练脚本

11. **`train_vtn_comparison.py`** ⭐⭐⭐⭐⭐
    ```bash
    # VTN 训练主脚本
    python train_vtn_comparison.py \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --crop_dim 224 \
        --pretrained_path ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
    ```

12. **`train_i3d_comparison.py`** ⭐⭐⭐⭐⭐
    ```bash
    # I3D 训练主脚本
    python train_i3d_comparison.py \
        --manual_annotations manual_annotations.json \
        --frame_dir /path/to/frames \
        --crop_dim 224
    ```

### 评估和对比

13. **`evaluate_per_video.py`** ⭐⭐⭐⭐⭐
    ```bash
    # 单独评估每个视频的性能
    python evaluate_per_video.py \
        --checkpoint ./vtn_outputs/best_model.pt \
        --model_type vtn \
        --manual_annotations manual_annotations.json
    ```

14. **`compare_results.py`** ⭐⭐⭐⭐⭐
    ```bash
    # 对比多个模型的结果
    python compare_results.py \
        --vtn_results vtn_outputs/evaluation_results.json \
        --i3d_results i3d_outputs/evaluation_results.json \
        --mdfed_results MD-FED/md_fed_outputs/stage3/evaluation_results.json
    ```

### 验证工具

15. **`verify_crop_strategy.py`** ⭐⭐⭐
    ```bash
    # 验证裁剪策略配置是否正确
    python verify_crop_strategy.py \
        --data_dir vtn_data \
        --frame_dir /path/to/frames
    ```

16. **`check_frame_size.py`** ⭐⭐⭐
    ```bash
    # 分析帧尺寸，推荐最佳配置
    python check_frame_size.py /path/to/frames
    ```

17. **`download_pretrained_vit.py`** ⭐⭐⭐
    ```bash
    # 下载 ViT 预训练权重
    python download_pretrained_vit.py \
        --model_size small \
        --save_dir models/
    ```

## 📖 阅读顺序

### 第一次使用

```
1. QUICKSTART_Comparison_Models.md (必读)
   ↓
2. 选择模型 (VTN 或 I3D)
   ↓
3. 阅读对应的详细文档
   - VTN: README_VTN_Comparison.md
   - I3D: README_I3D_Comparison.md
   ↓
4. README_Fair_Comparison.md (确保公平对比)
   ↓
5. 运行训练脚本
   ↓
6. 运行 compare_results.py (对比结果)
```

### VTN 专用路径

```
1. QUICKSTART_VTN.md
   ↓
2. README_VTN_Training_Strategy.md
   ↓
3. README_Fair_Comparison.md
   ↓
4. train_vtn_comparison.py
   ↓
5. compare_results.py
```

### I3D 专用路径

```
1. QUICKSTART_Comparison_Models.md
   ↓
2. README_I3D_Comparison.md
   ↓
3. README_Fair_Comparison.md
   ↓
4. train_i3d_comparison.py
   ↓
5. compare_results.py
```

## 🎯 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| **首次尝试** | VTN (small) | 最快，效果好 |
| **GPU 显存有限** | VTN (tiny) | 显存占用小 |
| **追求准确率** | MD-FED Stage 3 | 多模态，最强 |
| **经典方法** | I3D | 3D CNN，经典 |
| **速度优先** | VTN (small) | 训练最快 |

## 📊 模型对比

| 特性 | VTN | I3D | MD-FED Stage 3 |
|------|-----|-----|----------------|
| **架构** | ViT + Transformer | 3D CNN | ResNet + GCN + GRU |
| **输入** | RGB | RGB | RGB + Flow + Skeleton |
| **参数量** | ~22M (small) | ~12M | ~15M |
| **显存占用** | 中 | 高 | 中 |
| **训练速度** | 快 | 慢 | 中 |
| **预训练** | ImageNet | Kinetics | Kinetics + COCO |

## 🔬 完整实验流程

### 步骤 1: 准备数据

```bash
# 确保有以下文件/目录
ls -l manual_annotations.json
ls -l /path/to/frames/
ls -l elements.txt
```

### 步骤 2: 训练 VTN

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./vtn_outputs \
    --pretrained_path ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500
```

### 步骤 3: 训练 I3D

```bash
python train_i3d_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --save_dir ./i3d_outputs \
    --crop_dim 224 \
    --clip_len 96 \
    --batch_size 4 \
    --num_epochs 500
```

### 步骤 4: 训练 MD-FED Stage 3（基线）

```bash
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/frames \
    --flow_dir /path/to/flows \
    --save_dir ./MD-FED/md_fed_outputs/stage3
```

### 步骤 5: 评估所有模型

```bash
# 评估 VTN
python evaluate_per_video.py \
    --checkpoint ./vtn_outputs/best_model.pt \
    --model_type vtn \
    --manual_annotations manual_annotations.json

# 评估 I3D
python evaluate_per_video.py \
    --checkpoint ./i3d_outputs/best_model.pt \
    --model_type i3d \
    --manual_annotations manual_annotations.json

# 评估 MD-FED
python evaluate_per_video.py \
    --checkpoint ./MD-FED/md_fed_outputs/stage3/best_model.pt \
    --model_type mdfed \
    --manual_annotations manual_annotations.json
```

### 步骤 6: 对比结果

```bash
python compare_results.py \
    --vtn_results ./vtn_outputs/evaluation_results.json \
    --i3d_results ./i3d_outputs/evaluation_results.json \
    --mdfed_results ./MD-FED/md_fed_outputs/stage3/evaluation_results.json
```

## ❓ FAQ 快速链接

| 问题 | 在哪找答案 |
|------|----------|
| VTN 还是 I3D？ | QUICKSTART_Comparison_Models.md |
| VTN 怎么训练？ | QUICKSTART_VTN.md |
| I3D 怎么训练？ | README_I3D_Comparison.md |
| 为什么对比是公平的？ | README_Fair_Comparison.md |
| GPU 内存不足怎么办？ | QUICKSTART_Comparison_Models.md → 显存优化 |
| 预训练权重下载失败？ | FIX_HTTP_403_ERROR.md |
| 数据增强策略？ | README_Fair_Comparison.md |
| crop_dim 应该用多少？ | README_Crop_Dim_Explanation.md |

## 🎓 学习路径

### 路径 1: 快速对比（1天）

```
1. 读 QUICKSTART_Comparison_Models.md (20分钟)
2. 训练 VTN (3小时)
3. 训练 I3D (6小时)
4. 对比结果 (10分钟)
```

### 路径 2: 深入研究（3天）

```
Day 1:
  - 阅读所有文档 (3小时)
  - 理解代码实现 (3小时)

Day 2:
  - 训练 VTN (多配置) (8小时)
  
Day 3:
  - 训练 I3D (多配置) (8小时)
  - 结果分析 (2小时)
```

### 路径 3: 完整实验（1周）

```
Week 1:
  - 文档学习 (1天)
  - VTN 实验 (2天)
  - I3D 实验 (2天)
  - MD-FED 对比 (1天)
  - 论文撰写 (1天)
```

## 📝 更新日志

- **2026-02-12**: 创建完整的对比实验文档体系
  - 添加 VTN 训练脚本和文档
  - 添加 I3D 训练脚本和文档
  - 添加公平对比策略说明
  - 添加快速开始指南
  - 添加预训练权重下载工具
  - 添加评估和对比脚本

---

**从这里开始**: [QUICKSTART_Comparison_Models.md](QUICKSTART_Comparison_Models.md) 🚀

**VTN 专用**: [QUICKSTART_VTN.md](QUICKSTART_VTN.md)

**I3D 专用**: [README_I3D_Comparison.md](README_I3D_Comparison.md)
