# 训练和验证数据文件说明

## 数据流程

## 1. 原始数据文件位置

原始数据在 `F3Set/data/f3set-tennis/` 目录：
- `train.json` - 原始训练集
- `val.json` - 原始验证集  
- `test.json` - 原始测试集

## 2. 数据准备 (prepare_md_fed_data.py)

`prepare_md_fed_data.py` 会准备 MD-FED 训练所需的数据：

### 输出目录：`md_fed_data/f3set-tennis-sub/`

**训练集 (`train.json`)**:
- **来源**: 合并 `F3Set/data/f3set-tennis/train.json` + `F3Set/data/f3set-tennis/val.json`
- **目的**: 训练时使用更多数据
- **文件**: `md_fed_data/f3set-tennis-sub/train.json`

**验证集 (`val.json`)**:
- **默认情况** (`use_test_as_val=True`): 使用 `F3Set/data/f3set-tennis/test.json`
- **备选情况** (`use_test_as_val=False`): 使用 `F3Set/data/f3set-tennis/val.json`
- **文件**: `md_fed_data/f3set-tennis-sub/val.json`

## 3. 训练时使用的文件

在 `run_md_fed_stage1.py` 中：

```python
# 训练集
train_json = os.path.join(data_dir, args.dataset, 'train.json')
# 即: md_fed_data/f3set-tennis-sub/train.json
# 这是合并后的 train.json + val.json

# 验证集
val_json = os.path.join(data_dir, args.dataset, 'val.json')
# 即: md_fed_data/f3set-tennis-sub/val.json
# 默认是 test.json (如果 use_test_as_val=True)
```

## 总结

| 阶段 | 文件 | 来源 |
|------|------|------|
| **训练集** | `md_fed_data/f3set-tennis-sub/train.json` | `train.json` + `val.json` (合并) |
| **验证集** | `md_fed_data/f3set-tennis-sub/val.json` | `test.json` (默认) 或 `val.json` |

## 检查当前使用的文件

```bash
# 检查训练集
python -c "
import json
with open('md_fed_data/f3set-tennis-sub/train.json', 'r') as f:
    data = json.load(f)
print(f'Training set: {len(data)} videos')
"

# 检查验证集
python -c "
import json
with open('md_fed_data/f3set-tennis-sub/val.json', 'r') as f:
    data = json.load(f)
print(f'Validation set: {len(data)} videos')
# 显示第一个视频名，可以判断来源
if data:
    print(f'First video: {data[0].get(\"video\", \"unknown\")}')
"
```

## 重要提示

1. **训练集** 是合并后的数据（train + val），所以训练时使用了更多数据
2. **验证集** 默认使用 `test.json`，这样可以：
   - 保持原始 train/val 分离
   - 使用独立的测试集进行验证
3. 如果 F1 分数为 0，检查 `md_fed_data/f3set-tennis-sub/val.json` 是否正确加载
