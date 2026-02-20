# RGB alone as Student Ablation Study

## 实验目的

这个消融实验研究**只有 RGB 模态作为学生网络**学习 Skeleton 特征的效果，而 Flow 模态的参数被冻结（不更新）。

## 实验设计

### 原始 MD-FED Stage 2
- **教师网络**: Skeleton (Stage 1 预训练)
- **学生网络**: RGB, Flow
- **损失函数**: 
  - `MSE(RGB_feat, Skeleton_feat)`
  - `MSE(Flow_feat, Skeleton_feat)`

### 本消融实验 (RGB alone as Student)
- **教师网络**: Skeleton (Stage 1 预训练)
- **学生网络**: RGB only
- **Flow**: 参数冻结（`requires_grad=False`），不参与训练
- **损失函数**: 
  - `MSE(RGB_feat, Skeleton_feat)`

## 关键区别

1. **Flow 参数冻结**: Flow backbone 和 head 的所有参数都被设置为 `requires_grad=False`，在训练过程中不会更新
2. **只有 RGB 学习**: 只有 RGB 特征提取器会学习匹配 Skeleton 特征
3. **减少模态交互**: 移除了 Flow 和 Skeleton 之间的知识蒸馏

## 训练步骤

### Step 1: Stage 1 预训练（Skeleton）
```bash
python run_md_fed_stage1.py \
    --pose_dir /path/to/skeletons/f3set-tennis \
    --output_dir md_fed_outputs/stage1 \
    --num_epochs 500 \
    --batch_size 4 \
    --learning_rate 0.001
```

### Step 2: Stage 2 蒸馏（RGB alone as Student）
```bash
python train_stage2_rgb_student.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --flow_dir /path/to/ncaa_flow_rally \
    --pose_dir /path/to/ncaa_skeletons_rally \
    --stage1_model_dir ./md_fed_outputs/stage1 \
    --save_dir ./md_fed_outputs/stage2_rgb_student \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --clip_len 96 \
    --crop_dim 224
```

### Step 3: Stage 3 微调（Few-shot）
```bash
python few_shot_learning_stage3.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /path/to/ncaa_frames_rally \
    --flow_dir /path/to/ncaa_flow_rally \
    --stage2_model_dir ./md_fed_outputs/stage2_rgb_student \
    --save_dir ./md_fed_outputs/stage3_rgb_student \
    --batch_size 4 \
    --num_epochs 500 \
    --learning_rate 0.0001
```

## 预期结果

### 与原始 MD-FED 的对比

| 模型 | Stage 2 损失 | Stage 3 性能 |
|------|-------------|--------------|
| MD-FED (原始) | RGB→Skeleton + Flow→Skeleton | Baseline |
| RGB alone as Student | RGB→Skeleton only | 可能略低 |

### 预期影响

1. **性能下降**: 由于移除了 Flow 模态的学习，模型可能无法充分利用多模态信息
2. **特征质量**: RGB 特征可能不如原始 MD-FED 中 RGB 和 Flow 共同学习的效果好
3. **模态重要性**: 这个实验可以帮助评估 Flow 模态在知识蒸馏中的重要性

## 技术细节

### 参数冻结实现

```python
def _freeze_flow_parameters(self):
    """Freeze Flow backbone parameters so they don't get updated."""
    if hasattr(self._model, 'Impl'):
        model_impl = self._model.Impl
        if hasattr(model_impl, '_flow_backbone'):
            for param in model_impl._flow_backbone.parameters():
                param.requires_grad = False
        if hasattr(model_impl, '_flow_head'):
            for param in model_impl._flow_head.parameters():
                param.requires_grad = False
```

### 损失计算

```python
# Only RGB learns from Skeleton
rgb2sk_loss = F.mse_loss(rgb_feat, sk_feat)
loss = rgb2sk_loss
# Flow is frozen, so no flow2sk_loss
```

### 优化器

优化器只包含可训练的参数（RGB 和 Skeleton），Flow 参数被排除：

```python
trainable_params = [p for p in model._model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
```

## 输出文件

训练完成后，会在 `--save_dir` 目录下生成：

- `best_model.pt`: 最佳验证损失的模型
- `last_checkpoint.pt`: 最后一个 epoch 的检查点
- `config.json`: 训练配置（供 Stage 3 使用）

## 注意事项

1. **确保 Stage 1 完成**: 需要先完成 Skeleton 的 Stage 1 预训练
2. **数据准备**: 确保 RGB frames、optical flow 和 skeleton 数据都已准备好
3. **内存管理**: 如果遇到 CUDA OOM，可以减少 `batch_size` 或使用 `--acc_grad_iter`
4. **Skeleton 数据**: 使用 `collate_fn_skeleton_padding` 确保 skeleton 数据固定为 2 个人（单打网球）

## 相关消融实验

- **RGB as Teacher**: `train_stage2_rgb_teacher.py` - RGB 作为教师，Flow 和 Skeleton 学习 RGB
- **Flow as Teacher**: `train_stage2_flow_teacher.py` - Flow 作为教师，RGB 和 Skeleton 学习 Flow
- **RGB + Flow Fusion**: `train_rgb_flow_fusion.py` - RGB 和 Flow 融合，移除 Skeleton
