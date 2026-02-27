# F3ED 消融实验说明

本目录包含两个对比实验，用于研究迁移学习 vs 从头训练的效果。

## 📋 主模型架构（MD-FED 5阶段训练流程）

在介绍F3ED消融实验之前，先说明主模型架构（MD-FED）的完整训练流程：

### 主模型训练流程

#### 第一步：F3Set初步切割与预测
**目的**：对校园网球视频进行初步的事件检测和片段切割

**步骤**：
1. 使用F3Set预训练的F3ED模型对校园网球视频进行预测
2. 根据预测结果对视频进行初步切割，得到事件片段
3. 生成初步的标注结果（后续需要人工校验）

**输出**：
- 初步切割的视频片段
- 初步预测的标注结果（用于后续人工校验）

---

#### 第二步：HRNet抽取Skeleton
**目的**：基于F3Set切割出来的图片，使用HRNet抽取人体骨架数据

**步骤**：
1. 使用HRNet模型（`./deep-high-resolution-net.pytorch/models/pose_hrnet_w48_384x288.pth`）
2. 对第一步切割出来的图片进行骨架提取
3. 生成校园网球数据集的skeleton标注

**输出**：
- 校园网球数据集的skeleton标注文件

---

#### 第三步：MD-FED Stage-1（Teacher模型训练）
**目的**：在F3Set大赛网球的skeleton数据集上训练teacher模型

**关键点**：
- ✅ **使用F3Set大赛网球的skeleton数据集**（带标签）
- ❌ **不使用校园网球数据集**
- 训练STGCN++作为skeleton特征提取器（teacher模型）

**训练命令**：
```bash
python MD-FED/train_MD-FED.py f3set-tennis-sub \
    --pose_dir /path/to/f3set/skeletons \
    --stage 1 \
    --skeleton_arch stgcn++ \
    --save_dir ./md_fed_outputs/stage1
```

**输出**：
- Stage-1训练好的teacher模型（skeleton特征提取器）

---

#### 第四步：MD-FED Stage-2（无标签蒸馏）
**目的**：冻结teacher模型，使用校园网球的RGB+Flow数据集进行无标签蒸馏

**关键点**：
- ✅ **冻结Stage-1训练的teacher模型**（skeleton特征提取器）
- ✅ **使用校园网球的RGB+Flow数据集**（无标签）
- 训练RGB和Flow作为student模型，学习teacher模型的特征表示

**训练命令**：
```bash
python train_stage2_rgb_teacher.py \
    --frame_dir /path/to/college_tennis/frames \
    --flow_dir /path/to/college_tennis/flow \
    --pose_dir /path/to/college_tennis/skeletons \
    --stage1_model_dir ./md_fed_outputs/stage1 \
    --save_dir ./md_fed_outputs/stage2 \
    --batch_size 4 \
    --num_epochs 50
```

**输出**：
- Stage-2训练好的student模型（RGB+Flow特征提取器）

---

#### 第五步：MD-FED Stage-3（Few-shot微调）
**目的**：使用少量标注的校园网球数据集进行Few-shot预测

**关键点**：
- ✅ **使用少量标注的校园网球数据集**（第一步标注后人工校验的结果）
- 在Stage-2的基础上进行few-shot微调
- 比较Stage-3的预测结果与第一步（F3Set初步预测）的结果

**训练命令**：
```bash
python few_shot_learning_stage3.py \
    --stage2_checkpoint_dir ./md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/college_tennis/frames \
    --flow_dir /path/to/college_tennis/flow \
    --save_dir ./md_fed_outputs/stage3 \
    --num_epochs 500
```

**输出**：
- Stage-3最终模型
- 评估结果（与第一步F3Set预测结果对比）

---

### 主模型架构的优势

1. **多模态融合**：RGB + Flow + Skeleton三种模态
2. **知识蒸馏**：Skeleton作为teacher，指导RGB和Flow学习
3. **无标签学习**：Stage-2使用无标签的校园网球数据
4. **Few-shot适应**：Stage-3使用少量标注数据微调
5. **端到端优化**：从初步预测到最终精调的完整流程

---

## 🔬 F3ED消融实验（对比实验）

以下两个F3ED消融实验用于对比**单模态迁移学习**与**多模态蒸馏**的效果：

## 实验设计

### 实验1：F3ED在F3Set上训练，在校园网球的效果（迁移学习）

**目的**：评估F3Set预训练模型在校园网球数据集上的迁移效果

**方法**：
- 使用F3Set数据集上预训练的F3ED模型
- 直接在校园网球数据集上评估（**不进行微调**）
- 展示迁移学习的性能

**运行方式**：
```bash
python evaluate_f3ed_pretrained_on_college.py \
    --f3set_model_dir ./F3Set/f3set-model/f3ed \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --output_dir ./f3ed_pretrained_evaluation
```

**输出**：
- `evaluation_results.json`: 评估结果（包含Mean F1和Edit Score等指标）

---

### 实验2：F3ED不用F3Set训练，只在少量校园网球训练的效果（从头训练）

**目的**：评估从头训练F3ED模型在校园网球数据集上的效果

**方法**：
- **不进行F3Set预训练**
- 只在校园网球数据集上从头训练
- 使用与I3D/TSM等对比模型相同的数据划分（80% train, 20% val）
- 使用ImageNet预训练的backbone（但整个模型从头训练）

**运行方式**：
```bash
python train_f3ed_from_scratch_on_college.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./f3ed_from_scratch_outputs \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --feature_arch rny002_tsm \
    --temporal_arch gru \
    --use_ctx
```

**输出**：
- `checkpoint_*.pt`: 训练检查点
- `loss.json`: 训练历史
- `final_results.json`: 最终评估结果

---

## 对比分析

这两个实验可以回答以下问题：

1. **迁移学习是否有效？**
   - 实验1（迁移学习）vs 实验2（从头训练）
   - 如果实验1性能更好，说明F3Set预训练有助于校园网球任务

2. **预训练数据的重要性**
   - F3Set是大规模专业网球数据集
   - 校园网球是少量标注数据
   - 对比可以展示大规模预训练的价值

3. **与主模型（MD-FED）的对比**
   - **主模型（MD-FED）**：多模态蒸馏（RGB + Flow + Skeleton），5阶段训练
     - Stage-1: F3Set skeleton数据集训练teacher
     - Stage-2: 校园网球RGB+Flow无标签蒸馏
     - Stage-3: 校园网球少量标注few-shot微调
   - **F3ED实验1（迁移学习）**：单模态（RGB），F3Set预训练后直接评估
   - **F3ED实验2（从头训练）**：单模态（RGB），校园网球从头训练
   - 对比可以展示：
     - 多模态蒸馏 vs 单模态迁移学习的效果
     - 无标签蒸馏 vs 有标签训练的效果
     - 完整5阶段流程 vs 简单迁移学习的效果

## 评估指标

两个实验都使用相同的评估指标（与MD-FED一致）：

1. **Mean F1 (LCL)**: 事件定位F1分数
2. **Mean F1 (event)**: 事件级别F1分数
3. **Mean F1 (element)**: 元素级别F1分数
4. **Edit Score**: 编辑距离分数

## 数据划分

两个实验使用**相同的数据划分**（seed=42, 80% train, 20% val），确保公平对比。

## 预期结果

根据论文描述，预期结果：

- **实验1（迁移学习）**：应该比从头训练效果好，因为：
  - F3Set提供了大规模预训练数据
  - 模型已经学习了网球相关的特征

- **实验2（从头训练）**：性能可能较低，因为：
  - 只有少量校园网球标注数据
  - 没有利用F3Set的知识

- **与主模型（MD-FED）对比**：
  - **主模型（MD-FED）**：预期性能最好
    - 使用多模态蒸馏（RGB + Flow + Skeleton）
    - 5阶段训练流程，充分利用无标签数据
    - Stage-3 few-shot微调适应校园网球数据
  - **F3ED实验1（迁移学习）**：预期中等性能
    - 单模态（RGB），但有F3Set预训练
    - 直接评估，无微调
  - **F3ED实验2（从头训练）**：预期性能较低
    - 单模态（RGB），无预训练
    - 只有少量校园网球标注数据

## 注意事项

1. **模型架构一致性**：两个实验应使用相同的F3ED架构（feature_arch, temporal_arch, use_ctx）

2. **数据预处理一致性**：两个实验应使用相同的clip_len, crop_dim, stride等参数

3. **评估一致性**：两个实验应使用相同的评估函数和window参数

4. **元素类别一致性**：确保两个实验使用相同的elements.txt（类别定义）

## 文件说明

- `evaluate_f3ed_pretrained_on_college.py`: 实验1脚本（评估预训练模型）
- `train_f3ed_from_scratch_on_college.py`: 实验2脚本（从头训练）
- `README_F3ED_Ablation_Experiments.md`: 本说明文档

## 实验对比总结

| 实验 | 模态 | 预训练数据 | 训练数据 | 训练阶段 | 预期性能 |
|------|------|-----------|---------|---------|---------|
| **主模型（MD-FED）** | RGB+Flow+Skeleton | F3Set skeleton | 校园网球（无标签+少量标注） | 3阶段（Stage-1→2→3） | ⭐⭐⭐⭐⭐ 最高 |
| **F3ED实验1（迁移学习）** | RGB | F3Set | 校园网球（无标签，仅评估） | 0阶段（直接评估） | ⭐⭐⭐ 中等 |
| **F3ED实验2（从头训练）** | RGB | ImageNet | 校园网球（少量标注） | 1阶段（从头训练） | ⭐⭐ 较低 |

### 关键对比点

1. **模态数量**：
   - 主模型：3模态（RGB + Flow + Skeleton）
   - F3ED：1模态（RGB）

2. **训练策略**：
   - 主模型：知识蒸馏 + 无标签学习 + few-shot微调
   - F3ED实验1：直接迁移（无微调）
   - F3ED实验2：有监督学习（从头训练）

3. **数据利用**：
   - 主模型：充分利用无标签数据（Stage-2）和少量标注数据（Stage-3）
   - F3ED实验1：仅使用预训练知识
   - F3ED实验2：仅使用少量标注数据

## 相关文件

### 主模型（MD-FED）相关
- `MD-FED/train_MD-FED.py`: MD-FED主训练脚本
- `train_stage2_rgb_teacher.py`: Stage-2训练脚本（RGB作为teacher的消融）
- `train_stage2_flow_teacher.py`: Stage-2训练脚本（Flow作为teacher的消融）
- `few_shot_learning_stage3.py`: Stage-3 few-shot微调脚本
- `generate_skeleton_annotations.py`: HRNet骨架提取脚本

### F3ED消融实验相关
- `evaluate_f3ed_pretrained_on_college.py`: 实验1脚本（评估预训练模型）
- `train_f3ed_from_scratch_on_college.py`: 实验2脚本（从头训练）
- `F3Set/train_f3set_f3ed.py`: F3ED模型定义和训练代码
- `F3Set/test_f3set_f3ed.py`: F3ED评估代码

### 其他对比实验
- `train_i3d_comparison.py`: I3D对比实验（使用相同数据划分）
- `train_tsm_comparison.py`: TSM对比实验
- `train_vtn_comparison.py`: VTN对比实验
- `train_stgcn_comparison.py`: STGCN对比实验