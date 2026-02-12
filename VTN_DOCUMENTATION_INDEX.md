# VTN 文档索引

所有 VTN 相关的文档和工具汇总。

## 🚀 快速开始

### ⚠️ 最新更新：完整训练流程

0. **[README_VTN_Complete_Training.md](README_VTN_Complete_Training.md)** ⭐⭐⭐⭐⭐
   - **公平对比必读！**
   - 完整的 Stage 1 → Stage 3 训练流程
   - 模仿 MD-FED 的三阶段训练
   - 推荐用于正式对比实验

### 新手必读

1. **[VTN_STAGE1_QUICKSTART.md](VTN_STAGE1_QUICKSTART.md)** ⭐⭐⭐⭐⭐
   - **Stage 1 快速开始指南**
   - F3Set 数据预训练
   - 提供更好的初始化

2. **[QUICKSTART_VTN.md](QUICKSTART_VTN.md)** ⭐⭐⭐⭐
   - **Stage 3 快速测试**
   - 跳过 Stage 1 的快速验证
   - 适合快速原型

### 核心概念

3. **[README_VTN_Training_Strategy.md](README_VTN_Training_Strategy.md)** ⭐⭐⭐⭐⭐
   - VTN 应该如何训练？
   - 为什么与 MD-FED Stage 3 的对比是公平的？
   - 详细的训练策略说明

4. **[README_Fair_Comparison.md](README_Fair_Comparison.md)** ⭐⭐⭐⭐
   - 数据增强策略一致性
   - 训练时随机裁剪，评估时中心裁剪
   - 确保公平对比的关键配置

## 📚 详细指南

### 完整文档

4. **[README_VTN_Comparison.md](README_VTN_Comparison.md)** ⭐⭐⭐⭐
   - VTN 完整使用指南
   - 所有配置选项说明
   - 完整的对比流程

### 高级主题

5. **[README_Rectangular_Input.md](README_Rectangular_Input.md)** ⭐⭐⭐
   - 支持矩形输入 (384×224)
   - 保留更多原始帧信息
   - 不一定需要裁成正方形

6. **[README_Crop_Dim_Explanation.md](README_Crop_Dim_Explanation.md)** ⭐⭐⭐
   - crop_dim 参数详解
   - 为什么选择 112 或 224
   - 对小数据集的影响

7. **[README_Input_Data_Format.md](README_Input_Data_Format.md)** ⭐⭐
   - 输入数据格式说明
   - 模型期望的输入格式
   - 帧、光流、骨架数据

## 🛠️ 工具脚本

### 训练和评估

8. **`train_vtn_stage1.py`** ⭐⭐⭐⭐⭐ NEW!
   ```bash
   # VTN Stage 1: F3Set 预训练
   python train_vtn_stage1.py \
       --frame_dir /path/to/f3set_frames \
       --save_dir ./vtn_outputs/stage1_small \
       --crop_dim 224 \
       --vtn_spatial_size small
   ```

9. **`train_vtn_comparison.py`** ⭐⭐⭐⭐⭐
   ```bash
   # VTN Stage 3: Few-shot 微调
   python train_vtn_comparison.py \
       --manual_annotations manual_annotations.json \
       --frame_dir /path/to/frames \
       --stage1_checkpoint ./vtn_outputs/stage1_small/best_model.pt \
       --crop_dim 224
   ```

10. **`compare_results.py`** ⭐⭐⭐⭐⭐
    ```bash
    # 对比 VTN 和 MD-FED 结果
    python compare_results.py \
        --vtn_results vtn_outputs/comparison/best_model_metrics.json \
        --mdfed_results MD-FED/md_fed_outputs/stage3/evaluation_results.json
    ```

### 验证工具

10. **`verify_crop_strategy.py`** ⭐⭐⭐
    ```bash
    # 验证裁剪策略配置是否正确
    python verify_crop_strategy.py \
        --data_dir vtn_data \
        --frame_dir /path/to/frames
    ```

11. **`check_frame_size.py`** ⭐⭐⭐
    ```bash
    # 分析帧尺寸，推荐最佳配置
    python check_frame_size.py /path/to/frames
    ```

### 其他工具

12. **`evaluate_per_video.py`**
    ```bash
    # 单独评估每个视频的性能
    python evaluate_per_video.py \
        --checkpoint_dir vtn_outputs/comparison \
        --manual_annotations manual_annotations.json
    ```

## 📖 阅读顺序

### 正式对比实验 (推荐)

```
1. README_VTN_Complete_Training.md (理解完整流程)
   ↓
2. VTN_STAGE1_QUICKSTART.md (Stage 1 预训练)
   ↓
3. 运行 train_vtn_stage1.py (F3Set 预训练, 8-12h)
   ↓
4. 运行 train_vtn_comparison.py (Stage 3 微调, 2-4h)
   ↓
5. 运行 compare_results.py (对比结果)
```

### 快速测试 (跳过 Stage 1)

```
1. QUICKSTART_VTN.md (必读)
   ↓
2. README_VTN_Training_Strategy.md (理解训练策略)
   ↓
3. README_Fair_Comparison.md (确保公平对比)
   ↓
4. 运行 train_vtn_comparison.py (开始训练, 2-4h)
   ↓
5. 运行 compare_results.py (对比结果)
```

### 遇到问题

```
问题: GPU 内存不足？
→ 查看 QUICKSTART_VTN.md 的"常见问题"部分

问题: 不确定用哪个配置？
→ 查看 README_VTN_Comparison.md 的配置选项

问题: 想保留更多帧信息？
→ 查看 README_Rectangular_Input.md

问题: 训练策略是否公平？
→ 查看 README_VTN_Training_Strategy.md
```

### 深入研究

```
1. README_VTN_Comparison.md (完整指南)
2. README_Rectangular_Input.md (高级输入选项)
3. README_Crop_Dim_Explanation.md (参数调优)
4. 阅读代码注释
```

## 🎯 核心要点

### 最重要的 3 点

1. **VTN 需要训练，不是直接推理**
   - 使用 ImageNet 预训练 ViT
   - 在 manual_annotations.json 上训练
   - 不是直接用预训练模型推理

2. **与 MD-FED Stage 3 的对比是公平的**
   - 两者都使用预训练权重
   - 两者都在相同数据上训练
   - 数据增强策略完全一致

3. **推荐从 small 模型开始**
   - tiny: 快速验证 (5.7M 参数)
   - small: 标准对比 (22M 参数) ← 推荐
   - base: 最佳性能 (86M 参数)

## 📊 文档结构

```
VTN 文档
│
├── 快速开始
│   └── QUICKSTART_VTN.md ⭐⭐⭐⭐⭐
│
├── 核心概念
│   ├── README_VTN_Training_Strategy.md ⭐⭐⭐⭐⭐
│   └── README_Fair_Comparison.md ⭐⭐⭐⭐
│
├── 完整指南
│   └── README_VTN_Comparison.md ⭐⭐⭐⭐
│
└── 高级主题
    ├── README_Rectangular_Input.md ⭐⭐⭐
    ├── README_Crop_Dim_Explanation.md ⭐⭐⭐
    └── README_Input_Data_Format.md ⭐⭐
```

## 🔗 相关资源

### 论文

- Vision Transformer: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) (ICLR 2021)
- Longformer: ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150) (2020)
- VTN: "Video Transformer Network" (2021)

### 代码

- `MD-FED/model/vtn.py` - VTN 模型实现
- `train_vtn_comparison.py` - 训练脚本
- `MD-FED/dataset/input_process.py` - 数据加载

### 工具库

- [timm](https://github.com/rwightman/pytorch-image-models) - Vision Transformer 预训练模型
- [einops](https://github.com/arogozhnikov/einops) - 张量操作
- PyTorch - 深度学习框架

## ❓ FAQ 快速链接

| 问题 | 在哪找答案 |
|------|----------|
| VTN 怎么训练？ | QUICKSTART_VTN.md |
| 为什么对比是公平的？ | README_VTN_Training_Strategy.md |
| GPU 内存不足怎么办？ | QUICKSTART_VTN.md → 常见问题 |
| 如何保留更多帧信息？ | README_Rectangular_Input.md |
| 数据增强策略是什么？ | README_Fair_Comparison.md |
| 所有配置选项？ | README_VTN_Comparison.md |
| crop_dim 应该用多少？ | README_Crop_Dim_Explanation.md |
| 输入是什么格式？ | README_Input_Data_Format.md |

## 🎓 学习路径

### 路径 1: 快速测试（4小时）⚡

```
1. 读 QUICKSTART_VTN.md (10分钟)
2. 读 README_VTN_Training_Strategy.md (15分钟)
3. 运行 train_vtn_comparison.py (跳过 Stage 1, 3小时)
4. 对比结果 (5分钟)

⚠️ 注意: 性能可能较差，不适合正式对比
```

### 路径 2: 完整对比（12-16小时）⭐ 推荐

```
1. 读 README_VTN_Complete_Training.md (20分钟)
2. 读 VTN_STAGE1_QUICKSTART.md (15分钟)
3. 运行 train_vtn_stage1.py (F3Set 预训练, 8-12小时)
4. 运行 train_vtn_comparison.py (Stage 3 微调, 2-4小时)
5. 对比结果 (5分钟)

✅ 推荐: 这是公平对比的正确方法
```

### 路径 3: 深入研究（2天）🔬

```
Day 1:
1. 阅读所有文档 (3小时)
2. 理解代码实现 (3小时)
3. Stage 1 预训练实验 (8-12小时)

Day 2:
4. Stage 3 微调实验 (4小时)
5. 多配置对比 (4小时)
6. 结果分析和论文撰写 (4小时)
```

## 📝 更新日志

- **2026-02-12 (v2)**: 添加 Stage 1 预训练支持 ⭐ NEW
  - ✅ 新增 `train_vtn_stage1.py` - F3Set 预训练脚本
  - ✅ 新增 `README_VTN_Complete_Training.md` - 完整训练流程
  - ✅ 新增 `VTN_STAGE1_QUICKSTART.md` - Stage 1 快速开始
  - ✅ 更新 `train_vtn_comparison.py` - 支持从 Stage 1 加载
  - ✅ 更新 `QUICKSTART_VTN.md` - 区分完整流程和快速测试
  - 📚 完全模仿 MD-FED 的三阶段训练流程

- **2026-02-12 (v1)**: 创建完整的 VTN 文档体系
  - 添加训练策略说明
  - 添加公平对比配置
  - 添加快速开始指南
  - 添加矩形输入支持
  - 添加验证工具

---

**从这里开始**: [QUICKSTART_VTN.md](QUICKSTART_VTN.md) 🚀
