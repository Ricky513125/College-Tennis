# 检查点管理指南

## 检查点保存机制

### 文件结构

训练时，每个 epoch 会保存：
```
stage3/
├── checkpoint_000.pt     # epoch 0 的模型
├── checkpoint_001.pt     # epoch 1 的模型
├── ...
├── checkpoint_490.pt     # epoch 490 的模型（假设是最佳）
├── checkpoint_500.pt     # epoch 500 的模型
├── optim_000.pt          # epoch 0 的优化器状态
├── optim_001.pt
├── ...
├── loss.json             # 所有 epoch 的训练历史
└── config.json           # 训练配置
```

### 继续训练时

如果从 epoch 490 继续训练 200 个 epoch：
- **不会覆盖** `checkpoint_000.pt` 到 `checkpoint_500.pt`
- **会添加** `checkpoint_501.pt` 到 `checkpoint_690.pt`
- **会更新** `loss.json`（追加新记录）

## 管理策略

### 方案 1: 同一目录继续（推荐）

```bash
# 从最佳 epoch 继续
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_best \
    --num_epochs 200 \
    --learning_rate 0.00001
```

**结果**：
```
stage3/
├── checkpoint_000.pt ... checkpoint_500.pt  # 保留
├── checkpoint_501.pt ... checkpoint_690.pt  # 新增
└── loss.json  # 包含 epoch 0-690 的完整历史
```

**优点**：
- ✅ 完整的训练历史
- ✅ 可以比较所有 epoch
- ✅ 如果新训练效果不好，仍保留旧的最佳模型

### 方案 2: 先备份，然后同一目录

```bash
# 1. 备份当前最佳结果
cp ./MD-FED/md_fed_outputs/stage3/checkpoint_490.pt \
   ./MD-FED/md_fed_outputs/stage3_backup_epoch490_score49.21.pt
cp ./MD-FED/md_fed_outputs/stage3/loss.json \
   ./MD-FED/md_fed_outputs/stage3_backup_loss.json

# 2. 继续训练
python improve_stage3_training.py \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_best \
    ...
```

**优点**：
- ✅ 有备份文件
- ✅ 主目录保持完整历史
- ✅ 可以随时恢复到备份版本

### 方案 3: 使用新目录（适合实验）

```bash
# 实验性训练（例如不同学习率）
python improve_stage3_training.py \
    --save_dir ./MD-FED/md_fed_outputs/stage3_lr1e5_0202 \
    --resume_from_best \
    --learning_rate 0.00001 \
    ...

# 另一个实验（不同 epoch）
python improve_stage3_training.py \
    --save_dir ./MD-FED/md_fed_outputs/stage3_epoch100_0202 \
    --resume_from_best \
    --num_epochs 100 \
    ...
```

**优点**：
- ✅ 多个实验独立
- ✅ 原始训练结果完全不受影响
- ✅ 便于对比不同配置

**注意**：虽然目录不同，但模型会从 `stage3` 加载，epoch 编号会继续（例如从 490 继续）

## 磁盘空间管理

### 检查磁盘使用

```bash
# 查看目录大小
du -sh ./MD-FED/md_fed_outputs/stage3

# 查看检查点文件大小
ls -lh ./MD-FED/md_fed_outputs/stage3/checkpoint_*.pt | head -5
```

### 清理旧检查点

如果磁盘空间不足，可以删除中间的检查点，**只保留关键的**：

```bash
cd ./MD-FED/md_fed_outputs/stage3

# 删除非关键 epoch（保留每 50 个 epoch）
# ⚠️ 小心操作！先备份
for i in {1..499}; do
    if [ $((i % 50)) -ne 0 ]; then
        rm -f checkpoint_$(printf "%03d" $i).pt
        rm -f optim_$(printf "%03d" $i).pt
    fi
done

# 保留这些关键检查点：
# - checkpoint_490.pt (最佳模型)
# - checkpoint_500.pt (最后一个)
# - 以及最新的所有检查点
```

### 自动清理策略

创建清理脚本：

```python
# cleanup_old_checkpoints.py
import os
import argparse

def cleanup_checkpoints(save_dir, keep_best=True, keep_last=True, keep_every_n=50):
    """保留关键检查点，删除中间的"""
    
    # 读取 loss.json 找到最佳 epoch
    loss_file = os.path.join(save_dir, 'loss.json')
    if os.path.exists(loss_file):
        import json
        with open(loss_file) as f:
            losses = json.load(f)
        best_epoch = max(losses, key=lambda x: x.get('val_edit', 0))['epoch']
        last_epoch = max(e['epoch'] for e in losses)
    
    # 确定要保留的 epoch
    keep_epochs = set()
    if keep_best:
        keep_epochs.add(best_epoch)
    if keep_last:
        keep_epochs.add(last_epoch)
    
    # 保留每 N 个 epoch
    for i in range(0, last_epoch + 1, keep_every_n):
        keep_epochs.add(i)
    
    # 删除其他的
    deleted = 0
    for file in os.listdir(save_dir):
        if file.startswith('checkpoint_') or file.startswith('optim_'):
            epoch = int(file.split('_')[1].split('.')[0])
            if epoch not in keep_epochs:
                os.remove(os.path.join(save_dir, file))
                deleted += 1
    
    print(f"Deleted {deleted} files")
    print(f"Kept {len(keep_epochs)} checkpoints: {sorted(keep_epochs)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--keep_every_n', type=int, default=50)
    args = parser.parse_args()
    
    cleanup_checkpoints(args.save_dir, keep_every_n=args.keep_every_n)
```

使用：
```bash
python cleanup_old_checkpoints.py ./MD-FED/md_fed_outputs/stage3 --keep_every_n 50
```

## 推荐的完整工作流程

```bash
# 1. 检查当前状态
python check_training_status.py ./MD-FED/md_fed_outputs/stage3

# 2. 备份最佳模型（可选但推荐）
cp ./MD-FED/md_fed_outputs/stage3/checkpoint_490.pt \
   ./MD-FED/md_fed_outputs/best_models/stage3_epoch490_score49.21.pt

# 3. 从最佳 epoch 继续训练
python improve_stage3_training.py \
    --stage2_checkpoint_dir ./MD-FED/md_fed_outputs/stage2 \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --flow_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_optical_flow_rally \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_best \
    --num_epochs 200 \
    --learning_rate 0.00001 \
    --eval_frequency 5 \
    --early_stop_patience 30

# 4. 训练完成后检查新状态
python check_training_status.py ./MD-FED/md_fed_outputs/stage3

# 5. 如果需要，清理中间检查点
# python cleanup_old_checkpoints.py ./MD-FED/md_fed_outputs/stage3
```

## 常见问题

### Q: 继续训练会覆盖之前的检查点吗？

**A**: 不会！因为 epoch 编号不同，新的检查点会有新的文件名（例如 `checkpoint_501.pt`），不会覆盖 `checkpoint_490.pt`。

### Q: 如果新训练效果不好怎么办？

**A**: 没关系！旧的检查点仍然存在。查看 `loss.json` 找到最佳 epoch，使用那个检查点即可：
```bash
python check_training_status.py ./MD-FED/md_fed_outputs/stage3
# 会显示最佳 epoch 和路径
```

### Q: 我想做多个实验，怎么管理？

**A**: 使用不同的目录名称，建议包含关键参数：
```bash
./MD-FED/md_fed_outputs/stage3_lr1e5_epoch200_0202
./MD-FED/md_fed_outputs/stage3_lr5e6_epoch100_0202
```

### Q: 磁盘空间不够了怎么办？

**A**: 
1. 只保留最佳和最后的检查点
2. 使用 `cleanup_old_checkpoints.py` 脚本
3. 或者手动删除中间的检查点（但保留关键的）

### Q: 怎么恢复到某个特定的检查点？

**A**: 使用 `--resume_from_epoch` 参数：
```bash
python improve_stage3_training.py \
    --save_dir ./MD-FED/md_fed_outputs/stage3 \
    --resume_from_epoch 490 \
    ...
```
