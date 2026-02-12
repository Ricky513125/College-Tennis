#!/usr/bin/env python3
"""
手动下载 ViT 预训练权重
解决 HTTP 403 错误
"""

import os
import torch
import timm
from pathlib import Path


def download_vit_weights(model_name='vit_small_patch16_224', cache_dir=None):
    """
    手动下载 ViT 预训练权重
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
    """
    print(f"{'='*80}")
    print(f"下载 {model_name} 预训练权重")
    print(f"{'='*80}\n")
    
    # 设置缓存目录
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"缓存目录: {cache_dir}\n")
    
    # 方法 1: 尝试直接创建模型（可能使用镜像源）
    print("方法 1: 尝试使用 timm 下载...")
    try:
        # 设置环境变量使用国内镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        model = timm.create_model(model_name, pretrained=True)
        print(f"✅ 成功下载 {model_name}")
        
        # 保存权重
        checkpoint_path = os.path.join(cache_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ 权重已保存到: {checkpoint_path}\n")
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ 方法 1 失败: {e}\n")
    
    # 方法 2: 从 Hugging Face 镜像下载
    print("方法 2: 尝试从 Hugging Face 镜像下载...")
    try:
        from huggingface_hub import hf_hub_download
        
        # ViT-Small 的 Hugging Face 仓库
        repo_id = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
        filename = "pytorch_model.bin"
        
        print(f"从 {repo_id} 下载...")
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"✅ 成功下载到: {checkpoint_path}\n")
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ 方法 2 失败: {e}\n")
    
    # 方法 3: 提供手动下载链接
    print("方法 3: 手动下载")
    print(f"{'─'*80}")
    print("请手动下载预训练权重:")
    print()
    print("选项 1 - 从 timm GitHub:")
    print(f"  URL: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/{model_name}.pth")
    print(f"  保存到: {cache_dir}/{model_name}.pth")
    print()
    print("选项 2 - 从 Hugging Face (国内镜像):")
    print(f"  URL: https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin")
    print(f"  保存到: {cache_dir}/{model_name}.pth")
    print()
    print("下载后，重新运行训练脚本即可。")
    print(f"{'─'*80}\n")
    
    return None


def test_model_loading(model_name='vit_small_patch16_224'):
    """测试模型是否能正常加载"""
    print(f"{'='*80}")
    print("测试模型加载")
    print(f"{'='*80}\n")
    
    try:
        print(f"尝试加载 {model_name}...")
        model = timm.create_model(model_name, pretrained=True)
        print(f"✅ 模型加载成功！")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='下载 ViT 预训练权重'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vit_small_patch16_224',
        choices=['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224'],
        help='模型名称'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='缓存目录 (默认: ~/.cache/torch/hub/checkpoints)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='仅测试模型是否能加载'
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = test_model_loading(args.model)
        if success:
            print("\n🎉 模型可以正常使用！")
        else:
            print("\n⚠️  需要先下载预训练权重")
    else:
        download_vit_weights(args.model, args.cache_dir)


if __name__ == '__main__':
    main()
