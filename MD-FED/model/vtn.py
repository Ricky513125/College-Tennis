import torch
from torch import nn, einsum
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from model.longformer import Longformer
from model.linformer import Linformer
from model.transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms


class VTN(nn.Module):
    def __init__(self, frames, num_classes, img_size=224, img_height=None, img_width=None, 
                 patch_size=16, spatial_frozen=False, spatial_size='base', 
                 temporal_type='longformer', spatial_suffix='', pretrained=True, 
                 pretrained_path=None):
        super().__init__()
        self.frames = frames
        
        # 支持矩形输入
        if img_height is not None and img_width is not None:
            self.img_height = img_height
            self.img_width = img_width
            # timm 的 img_size 可以是 tuple (height, width)
            model_img_size = (img_height, img_width)
        else:
            self.img_height = img_size
            self.img_width = img_size
            model_img_size = img_size

        # # Convert args
        # spatial_args = Namespace(**spatial_args)
        # temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        # timm 支持矩形输入：img_size 可以是 int 或 (height, width) tuple
        
        # 如果提供了本地权重路径，直接从本地加载
        if pretrained_path is not None:
            import os
            if os.path.exists(pretrained_path):
                print(f"📂 从本地路径加载预训练权重: {pretrained_path}")
                # 先创建不带预训练的模型
                self.spatial_transformer = timm.create_model(
                    f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', 
                    pretrained=False,
                    img_size=model_img_size,
                    in_chans=3, 
                    attn_drop_rate=0.0, 
                    drop_rate=0.0
                )
                # 加载本地权重
                try:
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    # 处理可能的 state_dict 格式
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                    
                    self.spatial_transformer.load_state_dict(state_dict, strict=False)
                    print(f"✅ 成功从本地加载 ViT-{spatial_size} 预训练权重")
                except Exception as e:
                    print(f"⚠️  加载本地权重时出错: {e}")
                    print(f"⚠️  将使用随机初始化")
            else:
                print(f"⚠️  本地权重文件不存在: {pretrained_path}")
                print(f"⚠️  将使用随机初始化")
                self.spatial_transformer = timm.create_model(
                    f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', 
                    pretrained=False,
                    img_size=model_img_size,
                    in_chans=3, 
                    attn_drop_rate=0.0, 
                    drop_rate=0.0
                )
        else:
            # 尝试从网络下载
            try:
                self.spatial_transformer = timm.create_model(
                    f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', 
                    pretrained=pretrained,  # 使用传入的 pretrained 参数
                    img_size=model_img_size,  # 支持矩形
                    in_chans=3, 
                    attn_drop_rate=0.0, 
                    drop_rate=0.0
                )
                if pretrained:
                    print(f"✅ 成功加载 ViT-{spatial_size} 预训练权重")
                else:
                    print(f"⚠️  使用随机初始化的 ViT-{spatial_size}（未使用预训练）")
            except Exception as e:
                print(f"❌ 加载预训练权重失败: {e}")
                print(f"⚠️  回退到随机初始化...")
                self.spatial_transformer = timm.create_model(
                    f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', 
                    pretrained=False,  # 回退到不使用预训练
                    img_size=model_img_size,
                    in_chans=3, 
                    attn_drop_rate=0.0, 
                    drop_rate=0.0
                )
        
        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
          self.spatial_transformer.eval()
        # Spatial preprocess
        self.preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(img_size),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=self.spatial_transformer.default_cfg['mean'], std=self.spatial_transformer.default_cfg['std'])
        ])
        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.train_preprocess = create_transform(**config, is_training=True)

       
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)

        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # # Copy seq_len to frames
        # temporal_args.seq_len = frames
        
        if temporal_type == 'longformer':
          # self.temporal_transformer = Longformer(
          #   dim=768, depth=3, heads=12, dim_head=128, mlp_dim=3072, attention_window=8, attention_mode='sliding_chunks', emb_dropout=0.1, dropout=0.1, pool='cls', seq_len=frames)
          self.temporal_transformer = Longformer(
            dim=768, depth=1, heads=12, dim_head=128, mlp_dim=3072, attention_window=8, attention_mode='sliding_chunks', emb_dropout=0.1, dropout=0.1, pool='cls', seq_len=frames)
        elif temporal_type == 'linformer':
          self.temporal_transformer = Linformer(
            k=8, dim=768, depth=3, heads=12, dim_head=128, mlp_dim=3072, one_kv_head=True, share_kv=True, dropout=0.1, emb_dropout=0.5, seq_len=frames)
        elif temporal_type == 'transformer':
          # self.temporal_transformer = Transformer(
          #   dim=768, depth=3, heads=12, dim_head=128, mlp_dim=3072, dropout=0.1, seq_len=frames)
          self.temporal_transformer = Transformer(
              dim=192, depth=1, heads=12, dim_head=128, mlp_dim=1024, dropout=0.1, seq_len=frames)

        # # Classifer
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(temporal_args.dim),
        #     nn.Linear(temporal_args.dim, num_classes)
        # )
        # # Random init 0.0 mean, 0.02 std
        # nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)

    def forward(self, img):

        # x = self.collapse_frames(img)
        x = img
        
        # Spatial Transformer
        if self.spatial_frozen:
          with torch.no_grad():
            x = self.spatial_transformer.forward_features(x)
        else:
          x = self.spatial_transformer.forward_features(x)

        # print(x.shape)
  
        # Spatial to temporal
        x = self.spatial2temporal(x)

        # print(x.shape)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # print(x.shape)

        return x

        # print(x.shape)

        # # Classifier
        # return self.mlp_head(x)
