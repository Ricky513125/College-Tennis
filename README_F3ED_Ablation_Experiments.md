# F3ED 消融实验说明

本目录包含两个对比实验，用于研究迁移学习 vs 从头训练的效果。

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

3. **与MD-FED的对比**
   - MD-FED使用多模态蒸馏（RGB + Flow + Skeleton）
   - F3ED是单模态（RGB）模型
   - 可以对比单模态迁移学习 vs 多模态蒸馏的效果

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

- **与MD-FED对比**：
  - MD-FED使用多模态蒸馏，可能性能更好
  - F3ED是单模态模型，但如果有好的预训练，也可能表现不错

## 注意事项

1. **模型架构一致性**：两个实验应使用相同的F3ED架构（feature_arch, temporal_arch, use_ctx）

2. **数据预处理一致性**：两个实验应使用相同的clip_len, crop_dim, stride等参数

3. **评估一致性**：两个实验应使用相同的评估函数和window参数

4. **元素类别一致性**：确保两个实验使用相同的elements.txt（类别定义）

## 文件说明

- `evaluate_f3ed_pretrained_on_college.py`: 实验1脚本（评估预训练模型）
- `train_f3ed_from_scratch_on_college.py`: 实验2脚本（从头训练）
- `README_F3ED_Ablation_Experiments.md`: 本说明文档

## 相关文件

- `F3Set/train_f3set_f3ed.py`: F3ED模型定义和训练代码
- `F3Set/test_f3set_f3ed.py`: F3ED评估代码
- `train_i3d_comparison.py`: I3D对比实验（使用相同数据划分）
