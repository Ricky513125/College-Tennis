# 解决 HTTP 403 错误 - 下载预训练权重失败

## 🔥 问题

运行 `train_vtn_comparison.py` 时遇到错误：

```
urllib.error.HTTPError: HTTP Error 403: Forbidden
```

**原因**: 尝试从网络下载 ImageNet 预训练的 ViT 权重时被阻止（可能是网络、防火墙或服务器限制）。

## ✅ 解决方案

### 方案 1: 使用 --no_pretrained 选项（最快）

**临时跳过预训练权重，从随机初始化开始训练**：

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/no_pretrained \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50 \
    --no_pretrained  # ← 添加这个选项
```

**优点**：
- ✅ 立即可以运行
- ✅ 无需下载

**缺点**：
- ❌ 性能会降低（没有 ImageNet 预训练的帮助）
- ❌ 需要更多训练轮数才能收敛
- ❌ 对比不完全公平（MD-FED 使用了预训练）

**建议**: 先用这个方法验证代码能跑通，然后再想办法下载预训练权重。

### 方案 2: 手动下载预训练权重（推荐）

#### 步骤 1: 下载权重文件

**选项 A - 从 Hugging Face 镜像下载（国内推荐）**：

```bash
# 创建缓存目录
mkdir -p ~/.cache/torch/hub/checkpoints

# 下载 ViT-Small 权重（22M 参数）
wget https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -O models/vit_small_patch16_224.pth

# 或使用 curl
curl -L https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -o ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
```

**选项 B - 从 GitHub Releases 下载**：

```bash
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_patch16_224.pth \
    -O ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
```

**如果使用不同的 spatial_size**：

```bash
# Tiny (5.7M 参数)
wget https://hf-mirror.com/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -O ~/.cache/torch/hub/checkpoints/vit_tiny_patch16_224.pth

# Base (86M 参数)
wget https://hf-mirror.com/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -O ~/.cache/torch/hub/checkpoints/vit_base_patch16_224.pth
```

#### 步骤 2: 验证下载成功

```bash
ls -lh ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth

# 应该看到文件存在，大小约 80-90MB
```

#### 步骤 3: 重新运行训练

```bash
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

现在应该能成功加载预训练权重！

### 方案 3: 使用下载脚本

我提供了一个辅助脚本来尝试多种下载方法：

```bash
# 尝试下载 ViT-Small 权重
python download_pretrained_vit.py --model vit_small_patch16_224

# 测试模型是否能正常加载
python download_pretrained_vit.py --model vit_small_patch16_224 --test
```

如果看到 "✅ 模型加载成功"，说明权重已经可用！

### 方案 4: 配置代理（如果有）

如果你有代理服务器，可以配置环境变量：

```bash
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"

# 然后运行训练脚本
python train_vtn_comparison.py ...
```

### 方案 5: 使用国内镜像源

在 Python 脚本开始时设置镜像源：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

或在命令行设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train_vtn_comparison.py ...
```

## 📊 各方案对比

| 方案 | 速度 | 性能 | 推荐度 |
|------|------|------|--------|
| **方案 1: --no_pretrained** | ⚡⚡⚡ 立即可用 | ⭐⭐ 性能降低 | 🔧 临时方案 |
| **方案 2: 手动下载** | ⚡⚡ 需下载 80MB | ⭐⭐⭐⭐⭐ 完整性能 | ⭐⭐⭐⭐⭐ 推荐 |
| **方案 3: 下载脚本** | ⚡⚡ 需下载 | ⭐⭐⭐⭐⭐ 完整性能 | ⭐⭐⭐⭐ 推荐 |
| **方案 4: 代理** | ⚡⚡ 看网速 | ⭐⭐⭐⭐⭐ 完整性能 | ⭐⭐⭐ 如果有代理 |
| **方案 5: 镜像源** | ⚡⚡ 看网速 | ⭐⭐⭐⭐⭐ 完整性能 | ⭐⭐⭐⭐ 国内推荐 |

## 🎯 推荐流程

### 立即开始测试（使用方案 1）

```bash
# 先验证代码能跑通
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/test_no_pretrained \
    --crop_dim 224 \
    --vtn_spatial_size tiny \
    --num_epochs 10 \
    --no_pretrained  # 不使用预训练
```

### 正式训练（使用方案 2）

```bash
# 手动下载预训练权重
wget https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -O ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth

# 正式训练
python train_vtn_comparison.py \
    --manual_annotations manual_annotations.json \
    --frame_dir /mnt/ssd2/lingyu/College-Tennis/ncaa_frames_rally \
    --save_dir ./vtn_outputs/comparison \
    --crop_dim 224 \
    --vtn_spatial_size small \
    --num_epochs 50
```

## 🔍 验证是否成功

### 检查 1: 查看训练日志

成功使用预训练：
```
✅ 成功加载 ViT-small 预训练权重
```

未使用预训练：
```
⚠️  使用随机初始化的 ViT-small（未使用预训练）
```

### 检查 2: 查看权重文件

```bash
# 检查缓存目录
ls -lh ~/.cache/torch/hub/checkpoints/

# 应该看到 vit_small_patch16_224.pth (约 80-90MB)
```

### 检查 3: 对比训练效果

**使用预训练**：
- 初始 loss 较低 (~1.5-2.0)
- 快速收敛 (10-20 epochs)
- F1 分数较高

**不使用预训练**：
- 初始 loss 很高 (~3.0-5.0)
- 收敛慢 (50+ epochs)
- F1 分数较低（可能差 10-20%）

## ⚠️ 常见问题

### Q1: wget 命令不存在

```bash
# 使用 curl 代替
curl -L <URL> -o <output_file>
```

### Q2: 缓存目录不存在

```bash
# 创建目录
mkdir -p ~/.cache/torch/hub/checkpoints
```

### Q3: 下载很慢或中断

```bash
# 使用断点续传
wget -c <URL> -O <output_file>

# 或使用多线程下载工具
aria2c -x 16 <URL> -o <output_file>
```

### Q4: 文件损坏或无法加载

```bash
# 删除并重新下载
rm ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
wget <URL> -O ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
```

### Q5: 不同 spatial_size 需要不同权重

| spatial_size | 权重文件 | 大小 |
|--------------|---------|------|
| tiny | vit_tiny_patch16_224.pth | ~20MB |
| small | vit_small_patch16_224.pth | ~80MB |
| base | vit_base_patch16_224.pth | ~340MB |

## 📝 总结

**最快解决方案**: 使用 `--no_pretrained` 选项先跑通代码

```bash
python train_vtn_comparison.py ... --no_pretrained
```

**最佳解决方案**: 手动下载预训练权重

```bash
wget https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin \
    -O ~/.cache/torch/hub/checkpoints/vit_small_patch16_224.pth
```

**关键点**：
- ✅ 代码已经添加了自动回退机制（如果下载失败会自动使用随机初始化）
- ✅ 可以使用 `--no_pretrained` 选项明确跳过预训练
- ✅ 手动下载到正确的缓存目录即可解决问题

祝训练顺利！🚀🎾
